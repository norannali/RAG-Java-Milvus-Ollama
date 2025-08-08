package RAGApp;

import dev.langchain4j.model.ollama.OllamaLanguageModel;

public class OllamaTest {

    public static void main(String[] args) {

        OllamaLanguageModel model = OllamaLanguageModel.builder()
                .baseUrl("http://localhost:11434")
                .modelName("mistral")
                .build();

        String prompt = "What is Artificial Intelligence?";

        // 👇 generate the response first
        String response = model.generate(prompt).content();

        // 👇 then print the response
        System.out.println("🔍 Response from Ollama:");
        System.out.println(response);
    }
}
