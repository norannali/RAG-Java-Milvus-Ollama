package com.example;

import RAGApp.*;

import java.io.IOException;
import java.util.List;
import java.util.Scanner;

public class Main {
    public static void main(String[] args) {
        String modelName = "nomic-embed-text";
        String documentPath = "src/main/resources/document.txt";
        int embeddingDimension = 768;  // known output of nomic-embed-text
        String collectionName = "rag_collection_768";  // avoid collisions
        int chunkSize = 500;

        System.out.println("🚀 Initializing RAG Application...");

        Embedder embedder = null;
        MilvusVectorStore vectorStore = null;
        RagService ragService = null;

        try {
            // Initialize embedder
            System.out.println("📡 Loading embedding model: " + modelName);
            embedder = new Embedder(modelName, embeddingDimension);
            System.out.println("✅ Model loaded with dimension: " + embedder.getEmbeddingDimension());

            // Initialize vector store
            System.out.println("🗄 Connecting to Milvus...");
            vectorStore = new MilvusVectorStore(embeddingDimension, "localhost", 19530, collectionName);
            System.out.println("✅ Vector store initialized: " + collectionName);

            // Setup RAG service
            ragService = new RagService(vectorStore, embedder);

            // Load document
            System.out.println("📚 Reading document: " + documentPath);
            List<String> chunks = DocumentLoader.loadDocumentChunks(documentPath, chunkSize);
            System.out.println("✂ Chunked into " + chunks.size() + " parts (size " + chunkSize + ")");

            // Index chunks
            for (int i = 0; i < chunks.size(); i++) {
                ragService.indexText(chunks.get(i));
                System.out.printf("⚡ Indexed chunk %d/%d (%.1f%%)%n",
                        i + 1, chunks.size(), ((i + 1) * 100.0) / chunks.size());
            }

            System.out.println("\n✅ Indexing complete!");

            // Query loop
            System.out.println("\n🔍 Ready to answer questions (type 'exit' to quit):");

            Scanner scanner = new Scanner(System.in);
            while (true) {
                System.out.print("\n❓ Your question: ");
                String query = scanner.nextLine().trim();

                if (query.equalsIgnoreCase("exit") || query.equalsIgnoreCase("quit")) {
                    System.out.println("👋 Exiting. Thanks!");
                    break;
                }

                if (!query.isEmpty()) {
                    try {
                        long start = System.currentTimeMillis();
                        String response = ragService.ask(query);
                        long duration = System.currentTimeMillis() - start;

                        System.out.println("\n💡 Response (" + duration + " ms):");
                        System.out.println("─".repeat(50));
                        System.out.println(response);
                        System.out.println("─".repeat(50));
                    } catch (Exception e) {
                        System.err.println("❌ Failed to process query: " + e.getMessage());
                    }
                }
            }

        } catch (IOException e) {
            System.err.println("❌ Failed to load document: " + e.getMessage());
        } catch (Exception e) {
            System.err.println("💥 Fatal error: " + e.getMessage());
            e.printStackTrace();
        } finally {
            System.out.println("\n🧹 Cleaning up resources...");
            try {
                if (ragService != null) ragService.close();
                if (vectorStore != null) vectorStore.close();
            } catch (Exception e) {
                System.err.println("⚠ Warning during cleanup: " + e.getMessage());
            }
        }
    }
}