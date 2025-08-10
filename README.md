# RAG-Java-Milvus-Ollama

A Retrieval-Augmented Generation (RAG) application built with **Java**, powered by **Milvus** vector database and **Ollama** for LLM inference. This project enables intelligent document search and question answering by combining semantic search with LLM capabilities.

---

## 🚀 Features

- 📄 Chunking and embedding of documents
- 🔍 Vector similarity search using **Milvus**
- 🧠 LLM integration via **Ollama** (Mistral)
- 💬 Question answering over embedded context
- 🧱 Modular architecture using Java and LangChain4j

---

## 📁 Project Structure

RAG-Java-Milvus-Ollama/

├── src/

│ └── main/

│ └── java/

│ ├── RAGApp/

│ │ ├── RagLLM.java # LLM interaction (Ollama)

│ │ ├── MilvusVectorStore.java # Milvus client logic

│ │ ├── RagService.java # Core RAG pipeline

│ │ └── OllamaTest.java # LLM testing

│ └── com/

│ └── example/

│ └── Main.java # Entry point

├── volumes/ # Local data volumes for Milvus and MinIO

├── .gitignore

├── pom.xml # Maven dependencies

└── README.md


---

## ⚙️ Requirements

- **Java 21+**
- **Maven 3.8+**
- **Docker** (for running Milvus and MinIO locally)
- **Ollama** installed locally with a model pulled (e.g., `ollama run mistral`)

---

