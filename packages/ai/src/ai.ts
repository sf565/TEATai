import type {
  AIAdapter,
  ChatCompletionOptions,
  ChatCompletionResult,
  ChatCompletionChunk,
  StreamChunk,
  TextGenerationOptions,
  TextGenerationResult,
  SummarizationOptions,
  SummarizationResult,
  EmbeddingOptions,
  EmbeddingResult,
} from "./types";

export class AI {
  private adapter: AIAdapter;

  constructor(adapter: AIAdapter) {
    this.adapter = adapter;
  }

  /**
   * Get the current adapter name
   */
  get adapterName(): string {
    return this.adapter.name;
  }

  /**
   * Complete a chat conversation
   */
  async chat(options: ChatCompletionOptions): Promise<ChatCompletionResult> {
    return this.adapter.chatCompletion(options);
  }

  /**
   * Complete a chat conversation with streaming (legacy)
   * @deprecated Use streamChat() for structured streaming with JSON chunks
   */
  async *chatStream(
    options: ChatCompletionOptions
  ): AsyncIterable<ChatCompletionChunk> {
    yield* this.adapter.chatCompletionStream({ ...options, stream: true });
  }

  /**
   * Stream chat with structured JSON chunks (supports tools and detailed token info)
   * Automatically executes tools if they have execute functions
   */
  async *streamChat(
    options: ChatCompletionOptions
  ): AsyncIterable<StreamChunk> {
    const hasToolExecutors = options.tools?.some((t) => t.execute);

    // If no tool executors, just stream normally
    if (!hasToolExecutors) {
      yield* this.adapter.chatStream({ ...options, stream: true });
      return;
    }

    // Auto-execute tools
    const maxIterations = options.maxIterations ?? 5;
    const messages = [...options.messages];
    let iteration = 0;

    while (iteration < maxIterations) {
      iteration++;

      const toolCalls: import("./types").ToolCall[] = [];
      const toolCallsMap = new Map<
        number,
        { id: string; name: string; args: string }
      >();
      let hasToolCalls = false;

      // Stream the current iteration
      for await (const chunk of this.adapter.chatStream({
        ...options,
        messages,
        stream: true,
      })) {
        yield chunk;

        // Accumulate tool calls
        if (chunk.type === "tool_call") {
          const existing = toolCallsMap.get(chunk.index) || {
            id: chunk.toolCall.id,
            name: "",
            args: "",
          };

          if (chunk.toolCall.function.name) {
            existing.name = chunk.toolCall.function.name;
          }
          existing.args += chunk.toolCall.function.arguments;
          toolCallsMap.set(chunk.index, existing);
        }

        // Check if we need to execute tools
        if (chunk.type === "done" && chunk.finishReason === "tool_calls") {
          hasToolCalls = true;
          toolCallsMap.forEach((call) => {
            toolCalls.push({
              id: call.id,
              type: "function",
              function: {
                name: call.name,
                arguments: call.args,
              },
            });
          });
        }
      }

      // If no tool calls, we're done
      if (!hasToolCalls || toolCalls.length === 0) {
        break;
      }

      // Add assistant message with tool calls
      messages.push({
        role: "assistant",
        content: null,
        toolCalls,
      });

      // Execute tools
      for (const toolCall of toolCalls) {
        const tool = options.tools?.find(
          (t) => t.function.name === toolCall.function.name
        );

        if (tool?.execute) {
          try {
            const args = JSON.parse(toolCall.function.arguments);
            const result = await tool.execute(args);

            messages.push({
              role: "tool",
              content: result,
              toolCallId: toolCall.id,
              name: toolCall.function.name,
            });

            // Yield a custom chunk for tool execution
            yield {
              type: "content",
              id: this.generateId(),
              model: options.model,
              timestamp: Date.now(),
              delta: "",
              content: `[Tool ${toolCall.function.name} executed]`,
              role: "assistant",
            } as StreamChunk;
          } catch (error: any) {
            messages.push({
              role: "tool",
              content: JSON.stringify({ error: error.message }),
              toolCallId: toolCall.id,
              name: toolCall.function.name,
            });
          }
        }
      }

      // Continue loop to get final response
    }
  }

  private generateId(): string {
    return `ai-${Date.now()}-${Math.random().toString(36).substring(7)}`;
  }

  /**
   * Generate text from a prompt
   */
  async generateText(
    options: TextGenerationOptions
  ): Promise<TextGenerationResult> {
    return this.adapter.generateText(options);
  }

  /**
   * Generate text from a prompt with streaming
   */
  async *generateTextStream(
    options: TextGenerationOptions
  ): AsyncIterable<string> {
    yield* this.adapter.generateTextStream({ ...options, stream: true });
  }

  /**
   * Summarize text
   */
  async summarize(options: SummarizationOptions): Promise<SummarizationResult> {
    return this.adapter.summarize(options);
  }

  /**
   * Create embeddings for text
   */
  async embed(options: EmbeddingOptions): Promise<EmbeddingResult> {
    return this.adapter.createEmbeddings(options);
  }

  /**
   * Set a new adapter
   */
  setAdapter(adapter: AIAdapter): void {
    this.adapter = adapter;
  }
}
