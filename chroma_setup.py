from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

# Create dummy documents about various topics
dummy_documents = [
    Document(
        page_content="Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991.",
        metadata={"source": "programming", "topic": "python"}
    ),
    Document(
        page_content="Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.",
        metadata={"source": "ai", "topic": "machine_learning"}
    ),
    Document(
        page_content="LangGraph is a library for building stateful, multi-actor applications with LLMs. It extends LangChain by adding the ability to create cyclical graphs and manage complex agent workflows.",
        metadata={"source": "frameworks", "topic": "langgraph"}
    ),
    Document(
        page_content="Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with text generation. It retrieves relevant documents from a knowledge base and uses them to generate more accurate and contextual responses.",
        metadata={"source": "ai", "topic": "rag"}
    ),
    Document(
        page_content="Vector databases store data as high-dimensional vectors, enabling efficient similarity search. ChromaDB is an open-source vector database designed for AI applications, making it easy to build LLM apps with memory.",
        metadata={"source": "databases", "topic": "vector_db"}
    ),
    Document(
        page_content="OpenAI's GPT-4 is a large multimodal model that can accept both image and text inputs and produce text outputs. It exhibits human-level performance on various professional and academic benchmarks.",
        metadata={"source": "ai", "topic": "gpt4"}
    ),
]

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Create ChromaDB vector store
vectorstore = Chroma.from_documents(
    documents=dummy_documents,
    embedding=embeddings,
    collection_name="rag_knowledge_base",
    persist_directory="./chroma_db"
)

print("✅ ChromaDB initialized with dummy documents!")
print(f"📦 Vector store saved to: ./chroma_db")
print(f"📚 Total documents: {len(dummy_documents)}")