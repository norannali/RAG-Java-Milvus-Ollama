package RAGApp;

import java.util.List;
import java.util.Map;

public class RagService implements AutoCloseable {
    private final MilvusVectorStore vectorStore;
    private final Embedder embedder;
    private final RagLLM llm;

    public RagService(MilvusVectorStore vectorStore, Embedder embedder) {
        this.vectorStore = vectorStore;
        this.embedder = embedder;
        this.llm = new RagLLM();
    }

    // Index a text chunk by generating its embedding and storing it in Milvus
    public void indexText(String text) {
        float[] embedding = embedder.embed(text).vector();
        vectorStore.index(text, embedding);
    }

    // Ask a question using vector search + LLM answer generation
    public String ask(String question) {
        float[] queryEmbedding = embedder.embed(question).vector();
        Map<Long, String> topResults = vectorStore.search(queryEmbedding, 3);

        if (topResults.isEmpty()) {
            return "Sorry, I couldn't find relevant information.";
        }

        StringBuilder contextBuilder = new StringBuilder();
        for (Map.Entry<Long, String> entry : topResults.entrySet()) {
            contextBuilder.append("Chunk ID ").append(entry.getKey()).append(": ")
                    .append(entry.getValue()).append("\n\n");
        }

        return llm.generateAnswer(contextBuilder.toString(), question);
    }

    @Override
    public void close() {
        vectorStore.close();
    }
}
