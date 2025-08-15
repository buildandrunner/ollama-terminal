package main

import (
	"bufio"
	"context"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/ollama/ollama/api"
)

const (
	Reset  = "\033[0m"
	Green  = "\033[32m"
	Blue   = "\033[34m"
	Cyan   = "\033[36m"
	Yellow = "\033[33m"
	Red    = "\033[31m"
	Purple = "\033[35m"
)

func loadSystemMessage(filename string) (string, error) {
	data, err := os.ReadFile(filename)
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(data)), nil
}

func NewOllamaClient() *api.Client {
	client, err := api.ClientFromEnvironment()
	if err != nil {
		log.Fatalln(Red+"[ERROR]"+Reset, "Failed to create Ollama client:", err)
	}
	return client
}

func main() {
	ctx, cancel := context.WithTimeout(context.Background(), time.Second*5)
	defer cancel()

	client := NewOllamaClient()

	systemMsg, err := loadSystemMessage("system.txt")
	if err != nil {
		log.Printf("Could not load system message: %v", err)
		systemMsg = "You are a helpful assistant." // fallback
	}

	fmt.Println(Cyan + "🔌 Connecting to Ollama..." + Reset)
	if err := client.Heartbeat(ctx); err != nil {
		fmt.Fprintf(os.Stderr, "\n%s❌  OLLAMA CONNECTION FAILED%s\n", Red, Reset)
		fmt.Fprintf(os.Stderr, "────────────────────────────────────\n")
		fmt.Fprintf(os.Stderr, "📡  Could not reach Ollama at http://127.0.0.1:11434\n")
		fmt.Fprintf(os.Stderr, "💡  Tip: Start Ollama with: %sollama serve%s\n", Yellow, Reset)
		fmt.Fprintf(os.Stderr, "📦  Get Ollama: https://ollama.com/download\n")
		fmt.Fprintf(os.Stderr, "────────────────────────────────────\n\n")
		os.Exit(1)
	}
	fmt.Println(Green + "✅ Connected successfully!" + Reset)

	clientVersion, err := client.Version(ctx)
	if err != nil {
		log.Fatalln(Red+"[ERROR]"+Reset, "Failed to get version:", err)
	}
	fmt.Printf("%s📋 Client Version:%s %s\n\n", Yellow, Reset, clientVersion)

	listRes, err := client.List(ctx)
	if err != nil {
		log.Fatalln(Red+"[ERROR]"+Reset, "Failed to list models:", err)
	}

	defaultModel := "gpt-oss:20b"
	embeddingModel := "nomic-embed-text"

	fmt.Printf("%s📦 Available Models:%s\n", Yellow, Reset)
	for i, m := range listRes.Models {
		prefix := "  "
		if m.Name == defaultModel {
			prefix = "  " + Green + "★" + Reset + " "
		}
		fmt.Printf("%s%d: %s%s%s\n", prefix, i, Cyan, m.Name, Reset)
	}

	fmt.Printf("\n%s💬 Default Chat Model:%s %s\n", Yellow, Reset, defaultModel)
	fmt.Printf("%s🧩 Embedding Model:%s %s\n", Yellow, Reset, embeddingModel)

	// Show model capabilities
	showReq := &api.ShowRequest{Model: defaultModel}
	showRes, err := client.Show(ctx, showReq)
	if err != nil {
		log.Fatalln(Red+"[ERROR]"+Reset, "Show failed:", err)
	}
	fmt.Printf("\n%s⚙️  Capabilities of %s:%s\n", Yellow, defaultModel, Reset)
	for _, cap := range showRes.Capabilities {
		fmt.Printf("  - %s\n", cap)
	}

	// Chat loop
	reader := bufio.NewReader(os.Stdin)
	fmt.Println("\n" + Blue + "🗨️  Start chatting with your AI (type 'exit' to quit)" + Reset)

	// --- 🟢 New: Conversation History ---
	messages := make([]api.Message, 0)
	messages = append(messages, api.Message{
		Role:    "system",
		Content: systemMsg,
	})

	for {
		fmt.Print("\n" + Green + "📝 You: " + Reset)
		text, err := reader.ReadString('\n')
		if err != nil {
			// ... (error handling)
			continue
		}
		text = strings.TrimSpace(text)
		if text == "" {
			continue
		}
		if strings.ToLower(text) == "exit" || text == "quit" {
			fmt.Println(Blue + "👋 Goodbye! Stay safe." + Reset)
			break
		}

		// --- 🟢 New: Add the user's message to history ---
		messages = append(messages, api.Message{
			Role:    "user",
			Content: text,
		})

		longerCtx, cancel := context.WithTimeout(context.Background(), time.Second*30)
		// No defer cancel() here, it should be called at the end of the loop iteration

		var fullResponse strings.Builder
		thinkingDone := false
		think := &api.ThinkValue{Value: "low"}

		// --- 🟢 New: Use ChatRequest and Chat endpoint ---
		chatReq := &api.ChatRequest{
			Model:    defaultModel,
			Messages: messages, // Send the full message history
			Think:    think,
		}

		err = client.Chat(longerCtx, chatReq, func(resp api.ChatResponse) error {
			// --- Handle Thinking (optional, but good to keep) ---
			if resp.DoneReason == "" && resp.Message.Content == "" && !thinkingDone {
				// Your existing logic for thinking...
			}

			if resp.Message.Thinking != "" && !thinkingDone {
				// Your existing logic for finalizing thinking...
			}

			// --- Stream Response ---
			if resp.Message.Content != "" {
				fmt.Print(Blue + resp.Message.Content + Reset)
				fullResponse.WriteString(resp.Message.Content)
			}
			return nil
		})

		// 🟢 New: Add the model's response to history
		messages = append(messages, api.Message{
			Role:    "assistant",
			Content: fullResponse.String(),
		})

		if err != nil {
			fmt.Printf("\n%s❌ Generation failed:%s %v%s\n", Red, Reset, err, Reset)
			// Optional: you might want to remove the last user message from history on error
		}

		// Final newline after response
		fmt.Println()
		cancel() // Call cancel at the end of the loop
	}
}
