from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from duckduckgo_search import DDGS
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
import uuid

load_dotenv()

# Load vectorstore
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    collection_name="rag_knowledge_base",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    rag_result: str  # Add this field for storing RAG results

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

def answer(state: AgentState) -> AgentState:
    """Generate final answer using RAG context or search results"""
    rag_result = state.get("rag_result", "")
    query = state["messages"][-1].content
    
    # Create context-aware prompt
    prompt = f"Context: {rag_result}\n\nQuestion: {query}\n\nAnswer based on the context above:"
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [response]}

def rag_lookup(state: AgentState) -> AgentState:
    """Performs RAG lookup from ChromaDB"""
    query = state["messages"][-1].content
    
    # Search for relevant documents
    results = vectorstore.similarity_search(query, k=3)
    
    # Combine results into context
    context = "\n\n".join([doc.page_content for doc in results])
    
    # Store in state
    return {"rag_result": context}

def search_duckduckgo_node(state: AgentState) -> AgentState:
    """Node wrapper for DuckDuckGo search"""
    query = state["messages"][-1].content
    
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            
        if not results:
            search_result = "No results found."
        else:
            # Format results
            output = []
            for i, result in enumerate(results, 1):
                output.append(f"{i}. {result['title']}")
                output.append(f"   {result['body']}")
                output.append(f"   URL: {result['href']}\n")
            
            search_result = "\n".join(output)
    
    except Exception as e:
        search_result = f"Search failed: {str(e)}"
    
    # Add search results to RAG result
    return {"rag_result": state.get("rag_result", "") + "\n\n" + search_result}

def decision(state: AgentState) -> str:
    rag_result = state.get("rag_result", "")
    query = state["messages"][-1].content
    
    # First check: Basic length
    if not rag_result or len(rag_result.strip()) < 30:
        return "search_duckduckgo"
    
    # Second check: LLM judgment
    judge_prompt = f"""Query: {query}
Context: {rag_result[:300]}

Is this context sufficient to answer the query? Reply YES or NO."""
    
    judgment = llm.invoke([HumanMessage(content=judge_prompt)])
    
    if "NO" in judgment.content.upper():
        return "search_duckduckgo"
    
    return "answer"

def router(state: AgentState) -> str:
    """Route to greeting or RAG based on message type"""
    last_message = state["messages"][-1].content.lower()
    
    # Check for greetings
    greeting_keywords = ["hello", "hi", "hey", "greetings", "good morning", 
                        "good afternoon", "good evening", "howdy"]
    
    if any(keyword in last_message for keyword in greeting_keywords):
        return "greeting"
    
    return "rag_lookup"

def greeting(state: AgentState) -> AgentState:
    """Handle greeting messages"""
    response = llm.invoke([
        HumanMessage(content="Respond warmly to this greeting: " + state["messages"][-1].content)
    ])
    return {"messages": [response]}

# Build graph
graph = StateGraph(AgentState)

# Add nodes - REMOVED "router" node
graph.add_node("greeting", greeting)
graph.add_node("rag_lookup", rag_lookup)
graph.add_node("search_duckduckgo", search_duckduckgo_node)
graph.add_node("answer", answer)

# Add conditional edge directly from START
graph.add_conditional_edges(
    START,
    router,  # Use router as conditional function, not as a node
    {
        "greeting": "greeting",
        "rag_lookup": "rag_lookup"
    }
)

# Conditional routing from rag_lookup
graph.add_conditional_edges(
    "rag_lookup",
    decision,
    {
        "search_duckduckgo": "search_duckduckgo",
        "answer": "answer"
    }
)

# Edges to END
graph.add_edge("greeting", END)
graph.add_edge("search_duckduckgo", "answer")
graph.add_edge("answer", END)

# Initialize SQLite persistence
conn = sqlite3.connect("./checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)

# Compile graph with checkpointer
app = graph.compile(checkpointer=memory)

# Continuous chatbot loop with persistence
if __name__ == "__main__":
    print("=" * 60)
    print("RAG Agent Chatbot - Type 'quit', 'exit', or 'bye' to stop")
    print("=" * 60)
    
    # Create a unique thread ID for this conversation session
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    
    print(f"Session ID: {thread_id}")
    print("Your conversation is being saved to SQLite!")
    
    while True:
        # Get user input
        user_input = input("\nYou: ").strip()
        
        # Check for exit commands
        if user_input.lower() in ['quit', 'exit', 'bye', 'stop']:
            print("\nAgent: Goodbye! Have a great day!")
            print(f"Your conversation has been saved with session ID: {thread_id}")
            break
        
        # Skip empty inputs
        if not user_input:
            continue
        
        try:
            # Invoke the agent with persistence
            result = app.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config
            )
            
            # Print agent response
            agent_response = result["messages"][-1].content
            print(f"\nAgent: {agent_response}")
            
        except Exception as e:
            print(f"\nError: {str(e)}")
            print("Please try again.")