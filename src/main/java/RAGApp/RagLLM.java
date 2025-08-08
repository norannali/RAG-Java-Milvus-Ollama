package RAGApp;

import dev.langchain4j.model.ollama.OllamaLanguageModel;

import java.time.Duration;

public class RagLLM {

    private final OllamaLanguageModel llm;

    // Initialize with the lightweight "mistral" model
    public RagLLM() {
        this.llm = OllamaLanguageModel.builder()
                .baseUrl("http://localhost:11434")  // Required to avoid NullPointerException
                .modelName("mistral")               // Suitable model for limited-memory machines
                .timeout(Duration.ofSeconds(120))
                .build();
    }

    public String generateAnswer(String context, String question) {
        String prompt = String.format("""
                You are a helpful assistant. Use the following context to answer the user's question.

                Context:
                %s

                Question: %s

                Answer:""", context, question);

        return llm.generate(prompt).content();
    }
}
