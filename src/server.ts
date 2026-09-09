import { callable, routeAgentRequest, type Schedule } from "agents";
import { getSchedulePrompt, scheduleSchema } from "agents/schedule";
import { AIChatAgent, type OnChatMessageOptions } from "@cloudflare/ai-chat";
import { createWorkersAI } from "workers-ai-provider";
import {
  convertToModelMessages,
  pruneMessages,
  stepCountIs,
  streamText,
  tool,
} from "ai";
import { z } from "zod";

type WorkshopState = {
  memory: string[];
  projects: {
    id: string;
    name: string;
    description: string;
    status: "active" | "completed" | "blocked";
  }[];
};

type StreamTextOnFinish = NonNullable<
  Parameters<typeof streamText>[0]["onFinish"]
>;

const MAX_MEMORY_ENTRIES = 100;
const MAX_PROJECTS = 50;
const MAX_CONTEXT_ITEMS = 50;

export class ChatAgent extends AIChatAgent<Env> {
  initialState: WorkshopState = {
    memory: [],
    projects: [],
  };

  maxPersistedMessages = 100;
  chatRecovery = true;
  waitForMcpConnections = true;

  onStart() {
    this.mcp.configureOAuthCallback({
      customHandler: (result) => {
        if (result.authSuccess) {
          return new Response("<script>window.close();</script>", {
            headers: {
              "content-type": "text/html",
            },
          });
        }

        return new Response(
          `Authentication Failed: ${
            result.authError || "Unknown error"
          }`,
          {
            headers: {
              "content-type": "text/plain",
            },
            status: 400,
          },
        );
      },
    });
  }

  @callable()
  async addServer(name: string, url: string) {
    return await this.addMcpServer(name, url);
  }

  @callable()
  async removeServer(serverId: string) {
    await this.removeMcpServer(serverId);
  }

  async onChatMessage(
    onFinish: StreamTextOnFinish,
    options?: OnChatMessageOptions,
  ) {
    const workersai = createWorkersAI({
      binding: this.env.AI,
    });

    const mcpTools = this.mcp.getAITools();

    const workshopTools = {
      get_memory: tool({
        description:
          "Récupère uniquement la mémoire interne persistante.",
        inputSchema: z.object({}),
        execute: async () => ({
          source: "internal_memory",
          memories: this.state.memory.slice(-MAX_CONTEXT_ITEMS),
        }),
      }),

      save_memory: tool({
        description:
          "Enregistre une information dans la mémoire interne persistante.",
        inputSchema: z.object({
          content: z.string().min(1),
        }),
        execute: async ({ content }) => {
          const memory = [
            ...this.state.memory.filter(
              (item) => item !== content,
            ),
            content,
          ].slice(-MAX_MEMORY_ENTRIES);

          this.setState({
            ...this.state,
            memory,
          });

          return {
            success: true,
            source: "internal_memory",
            message: "Information mémorisée.",
            content,
          };
        },
      }),

      get_projects: tool({
        description:
          "Récupère uniquement les projets internes persistants.",
        inputSchema: z.object({}),
        execute: async () => ({
          source: "internal_projects",
          projects: this.state.projects.slice(-MAX_CONTEXT_ITEMS),
        }),
      }),

      save_project: tool({
        description:
          "Crée ou modifie un projet interne persistant.",
        inputSchema: z.object({
          id: z.string().min(1),
          name: z.string().min(1),
          description: z.string(),
          status: z.enum([
            "active",
            "completed",
            "blocked",
          ]),
        }),
        execute: async ({
          id,
          name,
          description,
          status,
        }) => {
          const projects = this.state.projects.filter(
            (project) => project.id !== id,
          );

          this.setState({
            ...this.state,
            projects: [
              ...projects,
              {
                id,
                name,
                description,
                status,
              },
            ].slice(-MAX_PROJECTS),
          });

          return {
            success: true,
            source: "internal_projects",
            message: `Projet "${name}" enregistré.`,
            project: {
              id,
              name,
              description,
              status,
            },
          };
        },
      }),

      list_mcp_servers: tool({
        description:
          "Liste uniquement les serveurs MCP connectés.",
        inputSchema: z.object({}),
        execute: async () => {
          const state = this.getMcpServers();

          return {
            source: "mcp_servers",
            servers: Object.entries(state.servers).map(
              ([id, server]) => ({
                id,
                name: server.name,
                state: server.state,
              }),
            ),
          };
        },
      }),
    };

    const activeProjects =
      this.state.projects
        .filter(
          (project) => project.status !== "completed",
        )
        .slice(-MAX_CONTEXT_ITEMS)
        .map(
          (project) =>
            `- ${project.name} | id=${project.id} | status=${project.status}`,
        )
        .join("\n") ||
      "Aucun projet actif enregistré.";

    const result = streamText({
      model: workersai("@cf/zai-org/glm-4.7-flash", {
        sessionAffinity: this.sessionAffinity,
      }),

      system: `Tu es l'ORCHESTRATEUR PERSONNEL de l'utilisateur.

Tu dois comprendre les demandes, utiliser les outils disponibles et agir réellement.

RÈGLES :

- Ne prétends jamais avoir effectué une action sans résultat positif de l'outil.
- Pour consulter la mémoire : utilise get_memory.
- Pour enregistrer une mémoire : utilise save_memory.
- Pour consulter les projets : utilise get_projects.
- Pour créer ou modifier un projet : utilise save_project.
- Pour consulter les connexions MCP : utilise list_mcp_servers.
- Ne mélange jamais mémoire, projets et MCP.
- Après l'exécution d'un outil, donne une réponse humaine claire.
- N'affiche jamais de balises XML de type <tool_call>.
- N'affiche jamais les paramètres internes des outils.
- Pour une tâche complexe, établis un plan court puis agis.

PROJETS ACTIFS :

${activeProjects}

Pour les données exactes, utilise get_projects.

${getSchedulePrompt({ date: new Date() })}`,

      messages: pruneMessages({
        messages: await convertToModelMessages(this.messages),
        toolCalls: "before-last-2-messages",
        reasoning: "before-last-message",
      }),

      tools: {
        ...mcpTools,
        ...workshopTools,

        getWeather: tool({
          description:
            "Obtient la météo actuelle d'une ville.",
          inputSchema: z.object({
            city: z.string(),
          }),
          execute: async ({ city }) => ({
            city,
            temperature:
              Math.floor(Math.random() * 30) + 5,
            condition: "sunny",
            unit: "celsius",
          }),
        }),

        getUserTimezone: tool({
          description:
            "Obtient le fuseau horaire du navigateur utilisateur.",
          inputSchema: z.object({}),
        }),

        calculate: tool({
          description:
            "Effectue un calcul mathématique.",
          inputSchema: z.object({
            a: z.number(),
            b: z.number(),
            operator: z.enum([
              "+",
              "-",
              "*",
              "/",
              "%",
            ]),
          }),
          execute: async ({
            a,
            b,
            operator,
          }) => {
            if (operator === "/" && b === 0) {
              return {
                error: "Division par zéro",
              };
            }

            const operations = {
              "+": a + b,
              "-": a - b,
              "*": a * b,
              "/": a / b,
              "%": a % b,
            };

            return {
              expression: `${a} ${operator} ${b}`,
              result: operations[operator],
            };
          },
        }),

        scheduleTask: tool({
          description:
            "Programme une tâche pour plus tard.",
          inputSchema: scheduleSchema,
          execute: async ({
            when,
            description,
          }) => {
            if (when.type === "no-schedule") {
              return "Planification invalide.";
            }

            const input =
              when.type === "scheduled"
                ? when.date
                : when.type === "delayed"
                  ? when.delayInSeconds
                  : when.type === "cron"
                    ? when.cron
                    : null;

            if (!input) {
              return "Type de planification invalide.";
            }

            try {
              this.schedule(
                input,
                "executeTask",
                description,
              );

              return `Tâche programmée : "${description}".`;
            } catch (error) {
              return `Erreur : ${error}`;
            }
          },
        }),

        getScheduledTasks: tool({
          description:
            "Liste les tâches programmées.",
          inputSchema: z.object({}),
          execute: async () => {
            const tasks = this.getSchedules();

            return tasks.length
              ? tasks
              : "Aucune tâche programmée.";
          },
        }),

        cancelScheduledTask: tool({
          description:
            "Annule une tâche programmée.",
          inputSchema: z.object({
            taskId: z.string(),
          }),
          execute: async ({ taskId }) => {
            try {
              this.cancelSchedule(taskId);
              return `Tâche ${taskId} annulée.`;
            } catch (error) {
              return `Erreur : ${error}`;
            }
          },
        }),
      },

      stopWhen: stepCountIs(12),

      abortSignal: options?.abortSignal,

      onFinish,
    });

    return result.toUIMessageStreamResponse();
  }

  async executeTask(
    description: string,
    _task: Schedule<string>,
  ) {
    console.log(
      `Executing scheduled task: ${description}`,
    );

    this.broadcast(
      JSON.stringify({
        type: "scheduled-task",
        description,
        timestamp: new Date().toISOString(),
      }),
    );
  }
}

export default {
  async fetch(
    request: Request,
    env: Env,
  ) {
    return (
      (await routeAgentRequest(request, env)) ||
      new Response("Not found", {
        status: 404,
      })
    );
  },
} satisfies ExportedHandler<Env>;