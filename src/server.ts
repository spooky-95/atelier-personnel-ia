import { createOpenAI } from "@ai-sdk/openai";
import { callable, routeAgentRequest, type Schedule } from "agents";
import { getSchedulePrompt, scheduleSchema } from "agents/schedule";
import { AIChatAgent, type OnChatMessageOptions } from "@cloudflare/ai-chat";
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
            headers: { "content-type": "text/html" },
            status: 200,
          });
        }
        return new Response(
          `Authentication Failed: ${result.authError || "Unknown error"}`,
          {
            headers: { "content-type": "text/plain" },
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
          "Récupère les informations actuellement enregistrées dans la mémoire persistante.",
        inputSchema: z.object({}),
        execute: async () => {
          return {
            memories: this.state.memory.slice(-MAX_CONTEXT_ITEMS),
          };
        },
      }),
      save_memory: tool({
        description:
          "Enregistre une information importante dans la mémoire persistante de l'utilisateur.",
        inputSchema: z.object({
          content: z.string().min(1),
        }),
        execute: async ({ content }) => {
          const existing = this.state.memory.filter(
            (item) => item !== content,
          );
          const memory = [...existing, content].slice(-MAX_MEMORY_ENTRIES);
          this.setState({
            ...this.state,
            memory,
          });
          return {
            success: true,
            message: "Information mémorisée.",
          };
        },
      }),
      get_projects: tool({
        description:
          "Liste les projets internes enregistrés. Utilise cet outil comme source de vérité avant de créer ou modifier un projet.",
        inputSchema: z.object({}),
        execute: async () => {
          return {
            projects: this.state.projects.slice(-MAX_CONTEXT_ITEMS),
          };
        },
      }),
      save_project: tool({
        description:
          "Crée ou met à jour un projet interne dans la mémoire persistante.",
        inputSchema: z.object({
          id: z.string().min(1),
          name: z.string().min(1),
          description: z.string(),
          status: z.enum(["active", "completed", "blocked"]),
        }),
        execute: async ({ id, name, description, status }) => {
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
            message: `Projet "${name}" enregistré.`,
          };
        },
      }),
      list_mcp_servers: tool({
        description:
          "Liste les serveurs MCP connectés et leur état. Permet de distinguer les ressources externes MCP des projets et de la mémoire interne.",
        inputSchema: z.object({}),
        execute: async () => {
          const state = this.getMcpServers();
          return {
            servers: Object.entries(state.servers).map(([id, server]) => ({
              id,
              name: server.name,
              state: server.state,
            })),
          };
        },
      }),
    };
    const openrouter = createOpenAI({
      apiKey: this.env.OPENROUTER_API_KEY,
      baseURL: "https://openrouter.ai/api/v1",
      headers: {
        "X-Title": "Atelier personnel IA",
      },
    });
    const activeProjects =
      this.state.projects
        .filter((project) => project.status !== "completed")
        .slice(-MAX_CONTEXT_ITEMS)
        .map(
          (project) =>
            `- [${project.status}] ${project.name} (id: ${project.id})`,
        )
        .join("\n") || "Aucun projet actif enregistré pour le moment.";
    const result = streamText({
      // Important :
      // .chat() force l'utilisation du endpoint Chat Completions,
      // compatible avec OpenRouter.
      model: openrouter.chat("z-ai/glm-5.2:free"),
      system: `Tu es l'ORCHESTRATEUR PERSONNEL de l'utilisateur.
Ton rôle n'est pas seulement de répondre : tu dois comprendre l'objectif, réfléchir aux étapes nécessaires et utiliser les outils disponibles lorsque cela est possible.
Pour chaque demande :
1. Comprends l'objectif réel.
2. Identifie le projet concerné.
3. Utilise get_projects lorsque le projet doit être identifié ou vérifié.
4. Détermine les tâches nécessaires.
5. Utilise les outils disponibles lorsque cela permet d'agir réellement.
6. Ne prétends jamais avoir effectué une action si elle n'a pas réellement été exécutée.
7. Vérifie le résultat lorsque c'est possible.
8. Signale clairement ce qui est terminé, en cours ou bloqué.
9. Si une information indispensable manque, pose une seule question précise.
10. Pour une demande complexe, commence par un plan court puis exécute la première étape possible.
11. Cherche toujours la solution la plus simple, gratuite et efficace.
Tu travailles comme un chef de projet technique autonome.
Tu dois privilégier l'ACTION plutôt que les longues explications.
MÉMOIRE :
- Si l'utilisateur demande explicitement de mémoriser une information, tu DOIS appeler save_memory.
- Ne dis jamais qu'une information est mémorisée si save_memory n'a pas réussi.
- Si l'utilisateur demande ce qui est mémorisé, tu DOIS appeler get_memory.
- Le résultat de get_memory est la source de vérité.
PROJETS :
- Pour consulter les projets, utilise get_projects.
- Pour créer ou modifier un projet, utilise save_project.
- Ne considère jamais une ressource MCP comme un projet interne.
- Ne mélange jamais les ressources MCP avec la mémoire ou les projets internes.
MCP :
- Les outils MCP sont des outils externes.
- Utilise-les lorsqu'ils permettent réellement d'effectuer l'action demandée.
- Si l'origine d'une ressource est ambiguë, utilise list_mcp_servers.
- Ne prétends jamais qu'un outil MCP a effectué une action si son appel n'a pas réussi.
ACTION :
Lorsqu'un outil permet de faire réellement quelque chose, utilise l'outil.
Ne remplace pas une action possible par une simple explication.
Format de suivi pour les projets complexes :
PROJET : [nom]
OBJECTIF :
[objectif]
PLAN :
1. [tâche]
2. [tâche]
3. [tâche]
ÉTAT :
[ce qui est terminé / en cours / bloqué]
PROCHAINE ACTION :
[action suivante]
N'utilise ce format complet que lorsque la demande est suffisamment complexe.
CONTEXTE DES PROJETS ACTIFS :
${activeProjects}
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
          description: "Get the current weather for a city",
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
            const temp = Math.floor(Math.random() * 30) + 5;
            return {
              city,
              temperature: temp,
              condition:
                conditions[
                  Math.floor(Math.random() * conditions.length)
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
            operator: z.enum(["+", "-", "*", "/", "%"]),
          }),
          needsApproval: async ({ a, b }) =>
            Math.abs(a) > 1000 || Math.abs(b) > 1000,
          execute: async ({ a, b, operator }) => {
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
          execute: async ({ when, description }) => {
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
      // 12 étapes maximum : suffisant pour les tâches complexes
      // tout en limitant la consommation du modèle gratuit.
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
  async fetch(request: Request, env: Env) {
    return (
      (await routeAgentRequest(request, env)) ||
      new Response("Not found", {
        status: 404,
      })
    );
  },
} satisfies ExportedHandler<Env>;