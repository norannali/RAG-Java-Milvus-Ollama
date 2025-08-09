package RAGApp;

import java.util.List;
import java.util.Map;

public class Retriever {

    private final MilvusVectorStore vectorStore;

    public Retriever(MilvusVectorStore vectorStore) {
        this.vectorStore = vectorStore;
    }

    /**
     * Retrieves the topK most similar vector IDs for the given query vector.
     *
     * @param queryVector The query embedding vector
     * @param topK        Number of similar results to retrieve
     * @return List of matching vector IDs
     */
    public Map<Long, String> retrieve(float[] queryVector, int topK) {
        return vectorStore.search(queryVector, topK);
    }
}
