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
		log.Fatalln(Red+"[ERROR]"+Reset, "Is Ollama running? Connect failed:", err)
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

	// Pull embedding model
	pullReq := &api.PullRequest{Model: embeddingModel}
	fmt.Printf("\n%s🔽 Pulling embedding model:%s %s...%s\n", Cyan, Reset, embeddingModel, Reset)
	pullProgressFn := func(r api.ProgressResponse) error {
		if r.Status == "success" {
			fmt.Printf("%s✅ Pulled model: %s%s\n", Green, r.Status, Reset)
		} else {
			fmt.Printf("   %s\n", r.Status)
		}
		return nil
	}
	if err := client.Pull(context.Background(), pullReq, pullProgressFn); err != nil {
		log.Fatalln(Red+"[ERROR]"+Reset, "Pull failed:", err)
	}

	// Test embedding
	embedInput := "Mary had a little lamb"
	embedReq := &api.EmbedRequest{
		Model: embeddingModel,
		Input: embedInput,
	}
	embedRes, err := client.Embed(ctx, embedReq)
	if err != nil {
		log.Fatalln(Red+"[ERROR]"+Reset, "Embedding failed:", err)
	}
	fmt.Printf("\n%s🧮 Generated embedding for:%s '%s' → %d dimensions\n", Purple, Reset, embedInput, len(embedRes.Embeddings[0]))

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

	for {
		fmt.Print("\n" + Green + "📝 You: " + Reset)
		text, err := reader.ReadString('\n')
		if err != nil {
			fmt.Println(Red+"❌ Error reading input:"+Reset, err)
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

		// Extended context for longer responses
		longerCtx, cancel := context.WithTimeout(context.Background(), time.Second*30)
		defer cancel()

		var fullResponse strings.Builder
		thinkingDone := false

		generateReq := &api.GenerateRequest{
			Model:  defaultModel,
			Prompt: text,
			System: systemMsg,
			Think:  &api.ThinkValue{Value: "low"}, // Set thinking level
		}

		err = client.Generate(longerCtx, generateReq, func(g api.GenerateResponse) error {
			// --- Handle Thinking (overwrite, don't append) ---
			if g.Thinking != "" && !thinkingDone {
				// ✅ Overwrite with the latest full thinking string
				currentThinking := g.Thinking

				// Optional: Truncate long text to avoid wrapping issues
				const maxLen = 100
				if len(currentThinking) > maxLen {
					currentThinking = currentThinking[:maxLen] + "..."
				}

				// ✅ Print on same line using \r and clear to end with \033[K
				fmt.Printf("\r%s🧠 Thinking...%s %s\033[K", Yellow, Reset, currentThinking)
			}

			// --- When Response Starts, Finalize Thinking ---
			if g.Response != "" && !thinkingDone {
				thinkingDone = true
				fmt.Printf("\r%s✅ Thought process complete.%s\033[K\n\n", Green, Reset)
			}

			// --- Stream Response ---
			if g.Response != "" {
				fmt.Print(Blue + g.Response + Reset)
				fullResponse.WriteString(g.Response)
			}

			return nil
		})

		if err != nil {
			fmt.Printf("\n%s❌ Generation failed:%s %v%s\n", Red, Reset, err, Reset)
			continue
		}

		// Final newline after response
		fmt.Println()
	}
}
