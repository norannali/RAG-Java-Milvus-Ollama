package RAGApp;

import dev.langchain4j.model.embedding.EmbeddingModel;
import dev.langchain4j.model.ollama.OllamaEmbeddingModel;
import dev.langchain4j.data.embedding.Embedding;

public class Embedder {
    private final EmbeddingModel embeddingModel;
    private final int embeddingDimension;

    public Embedder(String modelName) {
        this.embeddingModel = OllamaEmbeddingModel.builder()
                .baseUrl("http://localhost:11434")
                .modelName(modelName)
                .build();

        this.embeddingDimension = detectEmbeddingDimension();
        System.out.println("Auto-detected embedding dimension: " + this.embeddingDimension);
    }

    public Embedder(String modelName, int expectedDimension) {
        this.embeddingModel = OllamaEmbeddingModel.builder()
                .baseUrl("http://localhost:11434")
                .modelName(modelName)
                .build();

        int actualDimension = detectEmbeddingDimension();

        if (actualDimension != expectedDimension) {
            System.err.println("Warning: Expected dimension " + expectedDimension +
                    " but model produces " + actualDimension + " dimensions");
        }

        this.embeddingDimension = actualDimension;
    }

    private int detectEmbeddingDimension() {
        try {
            Embedding testEmbedding = embeddingModel.embed("test").content();
            return testEmbedding.vector().length;
        } catch (Exception e) {
            throw new RuntimeException("Failed to detect embedding dimension", e);
        }
    }

    public Embedding embed(String text) {
        if (text == null || text.trim().isEmpty()) {
            throw new IllegalArgumentException("Text cannot be null or empty");
        }
        return embeddingModel.embed(text).content();
    }

    public float[] embedAsFloatArray(String text) {
        return embed(text).vector();
    }

    public int getEmbeddingDimension() {
        return embeddingDimension;
    }
}
