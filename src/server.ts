import { createWorkersAI } from "workers-ai-provider";
import { callable, routeAgentRequest, type Schedule } from "agents";
import { getSchedulePrompt, scheduleSchema } from "agents/schedule";
import { AIChatAgent, type OnChatMessageOptions } from "@cloudflare/ai-chat";
import {
  convertToModelMessages,
  pruneMessages,
  stepCountIs,
  streamText,
  tool
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
export class ChatAgent extends AIChatAgent<Env> {
  initialState: WorkshopState = {
    memory: [],
    projects: [],
  };
maxPersistedMessages = 100;
  chatRecovery = true;
  // Wait for MCP connections to be re-established after hibernation before
  // processing a message, so MCP tools aren't intermittently missing.
  waitForMcpConnections = true;

  onStart() {
    // Configure OAuth popup behavior for MCP servers that require authentication
    this.mcp.configureOAuthCallback({
      customHandler: (result) => {
        if (result.authSuccess) {
          return new Response("<script>window.close();</script>", {
            headers: { "content-type": "text/html" },
            status: 200
          });
        }
        return new Response(
          `Authentication Failed: ${result.authError || "Unknown error"}`,
          { headers: { "content-type": "text/plain" }, status: 400 }
        );
      }
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

  async onChatMessage(_onFinish: unknown, options?: OnChatMessageOptions) {
    const mcpTools = this.mcp.getAITools();
    const workshopTools = {
    get_memory: tool({
    description:
      "Récupère les informations actuellement enregistrées dans la mémoire persistante.",
    inputSchema: z.object({}),
    execute: async () => {
      return {
        memories: this.state.memory,
      };
    },
  }),
save_memory: tool({
    description:
      "Enregistre une information importante dans la mémoire persistante de l'utilisateur.",
    inputSchema: z.object({
      content: z.string(),
    }),
    execute: async ({ content }) => {
      this.setState({
        ...this.state,
        memory: [...this.state.memory, content],
      });

      return {
        success: true,
        message: "Information mémorisée.",
      };
    },
  }),

  save_project: tool({
    description:
      "Crée ou met à jour un projet dans la mémoire persistante.",
    inputSchema: z.object({
      id: z.string(),
      name: z.string(),
      description: z.string(),
      status: z.enum(["active", "completed", "blocked"]),
    }),
    execute: async ({ id, name, description, status }) => {
      const projects = this.state.projects.filter((p) => p.id !== id);

      this.setState({
        ...this.state,
        projects: [
          ...projects,
          { id, name, description, status },
        ],
      });

      return {
        success: true,
        message: `Projet "${name}" enregistré.`,
      };
    },
  }),
};
const workersai = createWorkersAI({ binding: this.env.AI });

    const result = streamText({
      model: workersai("@cf/zai-org/glm-4.7-flash", {
        sessionAffinity: this.sessionAffinity
      }),
        system: `Tu es l’ORCHESTRATEUR PERSONNEL de l’utilisateur.

Ton rôle n’est pas seulement de répondre : tu dois comprendre l’objectif, réfléchir aux étapes nécessaires et utiliser les outils disponibles lorsque cela est possible.

Pour chaque demande :
1. Comprends l’objectif réel.
2. Identifie le projet concerné.
3. Détermine les tâches nécessaires.
4. Utilise les outils disponibles lorsque cela permet d’agir réellement.
5. Ne prétends jamais avoir effectué une action si elle n’a pas réellement été exécutée.
6. Vérifie le résultat lorsque c’est possible.
7. Signale clairement ce qui est terminé, en cours ou bloqué.
8. Si une information indispensable manque, pose une seule question précise.
9. Pour une demande complexe, commence par établir un plan court puis exécute la première étape possible.
10. Cherche toujours la solution la plus simple, gratuite et efficace.

Tu travailles comme un chef de projet technique autonome.
Tu dois privilégier l’ACTION au lieu de longues explications.

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

Ne donne ce format complet que lorsque la demande est suffisamment complexe pour le justifier.r questions about them.
MÉMOIRE ET PROJETS :

- Si l'utilisateur demande explicitement de mémoriser une information, tu DOIS appeler save_memory.
- Tu ne dois jamais dire qu'une information est mémorisée si save_memory n'a pas été exécuté avec succès.
- Si l'utilisateur demande ce qui est mémorisé, tu DOIS appeler get_memory.
- Tu dois utiliser le résultat de get_memory comme source de vérité, et non simplement le contexte de conversation.
- Pour créer ou modifier un projet, tu DOIS utiliser save_project.
${getSchedulePrompt({ date: new Date() })}

If the user asks to schedule a task, use the schedule tool to schedule the task.`,
      // Prune old tool calls and reasoning to save tokens on long conversations
      messages: pruneMessages({
        messages: await convertToModelMessages(this.messages),
        toolCalls: "before-last-2-messages",
        reasoning: "before-last-message"
      }),
      tools: {
        // MCP tools from connected servers
        ...mcpTools,
...workshopTools,

        // Server-side tool: runs automatically on the server
        getWeather: tool({
          description: "Get the current weather for a city",
          inputSchema: z.object({
            city: z.string().describe("City name")
          }),
          execute: async ({ city }) => {
            // Replace with a real weather API in production
            const conditions = ["sunny", "cloudy", "rainy", "snowy"];
            const temp = Math.floor(Math.random() * 30) + 5;
            return {
              city,
              temperature: temp,
              condition:
                conditions[Math.floor(Math.random() * conditions.length)],
              unit: "celsius"
            };
          }
        }),

        // Client-side tool: no execute function — the browser handles it
        getUserTimezone: tool({
          description:
            "Get the user's timezone from their browser. Use this when you need to know the user's local time.",
          inputSchema: z.object({})
        }),

        // Approval tool: requires user confirmation before executing
        calculate: tool({
          description:
            "Perform a math calculation with two numbers. Requires user approval for large numbers.",
          inputSchema: z.object({
            a: z.number().describe("First number"),
            b: z.number().describe("Second number"),
            operator: z
              .enum(["+", "-", "*", "/", "%"])
              .describe("Arithmetic operator")
          }),
          needsApproval: async ({ a, b }) =>
            Math.abs(a) > 1000 || Math.abs(b) > 1000,
          execute: async ({ a, b, operator }) => {
            const ops: Record<string, (x: number, y: number) => number> = {
              "+": (x, y) => x + y,
              "-": (x, y) => x - y,
              "*": (x, y) => x * y,
              "/": (x, y) => x / y,
              "%": (x, y) => x % y
            };
            if (operator === "/" && b === 0) {
              return { error: "Division by zero" };
            }
            return {
              expression: `${a} ${operator} ${b}`,
              result: ops[operator](a, b)
            };
          }
        }),

        scheduleTask: tool({
          description:
            "Schedule a task to be executed at a later time. Use this when the user asks to be reminded or wants something done later.",
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
            if (!input) return "Invalid schedule type";
            try {
              this.schedule(input, "executeTask", description, {
                idempotent: true
              });
              return `Task scheduled: "${description}" (${when.type}: ${input})`;
            } catch (error) {
              return `Error scheduling task: ${error}`;
            }
          }
        }),

        getScheduledTasks: tool({
          description: "List all tasks that have been scheduled",
          inputSchema: z.object({}),
          execute: async () => {
            const tasks = this.getSchedules();
            return tasks.length > 0 ? tasks : "No scheduled tasks found.";
          }
        }),

        cancelScheduledTask: tool({
          description: "Cancel a scheduled task by its ID",
          inputSchema: z.object({
            taskId: z.string().describe("The ID of the task to cancel")
          }),
          execute: async ({ taskId }) => {
            try {
              this.cancelSchedule(taskId);
              return `Task ${taskId} cancelled.`;
            } catch (error) {
              return `Error cancelling task: ${error}`;
            }
          }
        })
      },
      stopWhen: stepCountIs(20),
      abortSignal: options?.abortSignal
    });

    return result.toUIMessageStreamResponse();
  }

  async executeTask(description: string, _task: Schedule<string>) {
    // Do the actual work here (send email, call API, etc.)
    console.log(`Executing scheduled task: ${description}`);

    // Notify connected clients via a broadcast event.
    // We use broadcast() instead of saveMessages() to avoid injecting
    // into chat history — that would cause the AI to see the notification
    // as new context and potentially loop.
    this.broadcast(
      JSON.stringify({
        type: "scheduled-task",
        description,
        timestamp: new Date().toISOString()
      })
    );
  }
}

export default {
  async fetch(request: Request, env: Env) {
    return (
      (await routeAgentRequest(request, env)) ||
      new Response("Not found", { status: 404 })
    );
  }
} satisfies ExportedHandler<Env>;
