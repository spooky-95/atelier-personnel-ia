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
            status: 200,
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
    const mcpTools = this.mcp.getAITools();

    const workshopTools = {
      get_memory: tool({
        description:
          "SOURCE DE VÉRITÉ : mémoire interne persistante uniquement. " +
          "Utilise cet outil lorsque l'utilisateur demande ce qui est mémorisé, " +
          "ses souvenirs ou une information précédemment enregistrée. " +
          "Ne pas utiliser cet outil pour consulter les projets.",
        inputSchema: z.object({}),
        execute: async () => {
          return {
            source: "internal_memory",
            memories: this.state.memory.slice(-MAX_CONTEXT_ITEMS),
          };
        },
      }),

      save_memory: tool({
        description:
          "Écrit dans la mémoire interne persistante. " +
          "Utilise cet outil uniquement lorsque l'utilisateur demande explicitement " +
          "de mémoriser ou d'enregistrer une information.",
        inputSchema: z.object({
          content: z.string().min(1),
        }),
        execute: async ({ content }) => {
          const existing = this.state.memory.filter(
            (item) => item !== content,
          );

          const memory = [...existing, content].slice(
            -MAX_MEMORY_ENTRIES,
          );

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
          "SOURCE DE VÉRITÉ : projets internes uniquement. " +
          "Utilise obligatoirement cet outil lorsque l'utilisateur demande " +
          "la liste de ses projets, le nom d'un projet, sa description ou son statut. " +
          "Ne pas utiliser get_memory pour consulter les projets. " +
          "Ne pas utiliser un outil MCP pour consulter les projets internes.",
        inputSchema: z.object({}),
        execute: async () => {
          return {
            source: "internal_projects",
            projects: this.state.projects.slice(-MAX_CONTEXT_ITEMS),
          };
        },
      }),

      save_project: tool({
        description:
          "Écrit dans les projets internes persistants. " +
          "Utilise cet outil pour créer ou modifier un projet interne.",
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

          const updatedProjects = [
            ...projects,
            {
              id,
              name,
              description,
              status,
            },
          ].slice(-MAX_PROJECTS);

          this.setState({
            ...this.state,
            projects: updatedProjects,
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
          "SOURCE DE VÉRITÉ : connexions MCP externes uniquement. " +
          "Utilise cet outil pour connaître les serveurs MCP connectés et leur état.",
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

    /*
     * ============================================================
     * CLOUDFLARE WORKERS AI
     * ============================================================
     */

    const workersai = createWorkersAI({
      binding: this.env.AI,
    });

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

TON RÔLE :

Tu es le chef de projet technique de l'Atelier personnel IA.

Ton objectif est de comprendre une demande, déterminer ce qui doit être fait,
utiliser les outils disponibles et rendre compte honnêtement du résultat.

RÈGLES ABSOLUES :

1. Privilégie l'ACTION réelle.
2. Utilise les outils disponibles lorsqu'ils permettent réellement d'agir.
3. Ne prétends jamais avoir effectué une action si l'outil n'a pas réussi.
4. Ne transforme jamais une information supposée en fait.
5. Si une donnée persistante est demandée, utilise l'outil correspondant.
6. Ne mélange jamais mémoire interne, projets internes et ressources MCP.
7. Si un outil échoue, indique clairement l'échec.
8. Pour une tâche complexe, travaille étape par étape.
9. Vérifie le résultat lorsque cela est possible.
10. Pose une seule question uniquement lorsqu'une information indispensable manque.

MÉMOIRE :

Si l'utilisateur demande explicitement de mémoriser une information :
1. appelle save_memory ;
2. vérifie que success=true ;
3. seulement ensuite confirme la mémorisation.

Si save_memory échoue, ne dis jamais que l'information est mémorisée.

Si l'utilisateur demande ce qui est mémorisé :
- utilise obligatoirement get_memory ;
- utilise uniquement les données retournées par cet outil.

PROJETS :

Si l'utilisateur demande :
- quels sont ses projets ;
- la liste de ses projets ;
- le nom d'un projet ;
- la description d'un projet ;
- le statut d'un projet ;

utilise obligatoirement get_projects.

Pour créer ou modifier un projet :
1. utilise save_project ;
2. vérifie success=true ;
3. confirme uniquement après réussite.

Ne jamais utiliser get_memory pour répondre à une question sur les projets.

Ne jamais utiliser un outil MCP pour remplacer get_projects.

MCP :

Les outils MCP sont des outils externes connectés à l'Atelier.

Lorsqu'un outil MCP permet réellement d'effectuer une action :
- utilise-le ;
- attends son résultat ;
- ne prétends pas que l'action est terminée avant d'avoir reçu un résultat positif.

CONTEXTE ACTUEL DES PROJETS ACTIFS :

${activeProjects}

Ce contexte est uniquement un aperçu.
Pour obtenir les données exactes des projets, utilise get_projects.

FORMAT POUR LES PROJETS COMPLEXES :

PROJET : [nom]

OBJECTIF :
[objectif]

PLAN :
1. [tâche]
2. [tâche]
3. [tâche]

ÉTAT :
[terminé / en cours / bloqué]

PROCHAINE ACTION :
[action suivante]

Utilise ce format uniquement lorsque la demande est suffisamment complexe.

${getSchedulePrompt({ date: new Date() })}

Si l'utilisateur demande de programmer une tâche, utilise l'outil de planification.`,

      messages: pruneMessages({
        messages: await convertToModelMessages(this.messages),
        toolCalls: "before-last-2-messages",
        reasoning: "before-last-message",
      }),

      tools: {
        ...mcpTools,
        ...workshopTools,

        getWeather: tool({
          description: "Get the current weather for a city.",
          inputSchema: z.object({
            city: z.string(),
          }),
          execute: async ({ city }) => {
            const conditions = [
              "sunny",
              "cloudy",
              "rainy",
              "snowy",
            ];

            const temp =
              Math.floor(Math.random() * 30) + 5;

            return {
              city,
              temperature: temp,
              condition:
                conditions[
                  Math.floor(
                    Math.random() * conditions.length,
                  )
                ],
              unit: "celsius",
            };
          },
        }),

        getUserTimezone: tool({
          description:
            "Get the user's timezone from their browser.",
          inputSchema: z.object({}),
        }),

        calculate: tool({
          description:
            "Perform a math calculation with two numbers.",
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

          needsApproval: async ({ a, b }) =>
            Math.abs(a) > 1000 ||
            Math.abs(b) > 1000,

          execute: async ({
            a,
            b,
            operator,
          }) => {
            const ops: Record<
              string,
              (x: number, y: number) => number
            > = {
              "+": (x, y) => x + y,
              "-": (x, y) => x - y,
              "*": (x, y) => x * y,
              "/": (x, y) => x / y,
              "%": (x, y) => x % y,
            };

            if (operator === "/" && b === 0) {
              return {
                error: "Division by zero",
              };
            }

            return {
              expression: `${a} ${operator} ${b}`,
              result: ops[operator](a, b),
            };
          },
        }),

        scheduleTask: tool({
          description:
            "Schedule a task to be executed at a later time.",
          inputSchema: scheduleSchema,

          execute: async ({
            when,
            description,
          }) => {
            if (when.type === "no-schedule") {
              return "Not a valid schedule input";
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
              return "Invalid schedule type";
            }

            try {
              this.schedule(
                input,
                "executeTask",
                description,
              );

              return `Task scheduled: "${description}" (${when.type}: ${input})`;
            } catch (error) {
              return `Error scheduling task: ${error}`;
            }
          },
        }),

        getScheduledTasks: tool({
          description:
            "List all tasks that have been scheduled.",
          inputSchema: z.object({}),

          execute: async () => {
            const tasks = this.getSchedules();

            return tasks.length > 0
              ? tasks
              : "No scheduled tasks found.";
          },
        }),

        cancelScheduledTask: tool({
          description:
            "Cancel a scheduled task by its ID.",
          inputSchema: z.object({
            taskId: z.string(),
          }),

          execute: async ({ taskId }) => {
            try {
              this.cancelSchedule(taskId);

              return `Task ${taskId} cancelled.`;
            } catch (error) {
              return `Error cancelling task: ${error}`;
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