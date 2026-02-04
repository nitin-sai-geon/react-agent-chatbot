from typing import Annotated, TypedDict
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from tavily import TavilyClient
import sqlite3
import uuid
import random
import os
from datetime import datetime, timedelta

load_dotenv()

# Initialize Tavily client - use environment variable
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    collection_name="rag_knowledge_base",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    agent_result: str
    is_blocked: bool

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

# ========== LLM NODE (Prompt Protection) ==========
def llm_node(state: AgentState) -> AgentState:
    """Preprocessing and prompt protection"""
    user_message = state["messages"][-1].content.lower()
    
    blocked_keywords = [
        "system prompt", "ignore previous", "ignore instructions",
        "you are now", "forget everything", "system:", "assistant:",
        "reveal your prompt", "show me your instructions",
        "disregard", "override", "bypass", "jailbreak"
    ]
    
    if any(keyword in user_message for keyword in blocked_keywords):
        return {
        "messages": state["messages"] + [
            AIMessage(content="I cannot process that request.")
        ],
        "is_blocked": True
}

    
    return {"is_blocked": False}

# ========== ROUTING FUNCTION FOR LLM NODE ==========
def route_after_llm(state: AgentState) -> str:
    """Route based on whether prompt was blocked"""
    if state.get("is_blocked", False):
        return "blocked"
    return "continue"

# ========== SUPERVISOR NODE ==========
def supervisor(state: AgentState) -> str:
    """Supervisor decides which agent to route to"""
    query = state["messages"][-1].content
    
    prompt = f"""You are a supervisor routing user queries to specialized agents.

Available agents:
- booking: Hotel and flight booking requests
- weather: Weather forecasts and conditions
- math: Mathematical calculations and problem solving
- online_lookup: General knowledge questions, current events, web searches, time queries

User query: "{query}"

Analyze the query and respond with ONLY ONE WORD - the agent name (booking, weather, math, or online_lookup).
Examples:
- "Book a hotel in Paris" -> booking
- "What's the weather in Tokyo?" -> weather
- "Calculate 15% of 250" -> math
- "Who won the 2024 Olympics?" -> online_lookup
- "What time is it in London?" -> online_lookup"""
    
    decision = llm.invoke([HumanMessage(content=prompt)])
    agent_choice = decision.content.strip().lower()
    
    valid_agents = ["booking", "weather", "math", "online_lookup"]
    if agent_choice in valid_agents:
        return agent_choice
    return "online_lookup"


def booking_agent(state: AgentState) -> AgentState:
    """Simulate hotel and flight bookings"""
    query = state["messages"][-1].content.lower()
    
    if "hotel" in query:
        hotels = [
            {"name": "Grand Plaza Hotel", "price": "$150/night", "rating": "4.5/5", "location": "City Center"},
            {"name": "Seaside Resort", "price": "$200/night", "rating": "4.8/5", "location": "Beach Front"},
            {"name": "Budget Inn", "price": "$80/night", "rating": "4.0/5", "location": "Airport Area"}
        ]
        
        selected = random.choice(hotels)
        booking_id = f"HTL-{random.randint(10000, 99999)}"
        
        result = f"""Hotel Booking Simulated Successfully!

Booking Details:
- Booking ID: {booking_id}
- Hotel: {selected['name']}
- Location: {selected['location']}
- Price: {selected['price']}
- Rating: {selected['rating']}
- Check-in: {(datetime.now() + timedelta(days=7)).strftime('%B %d, %Y')}
- Check-out: {(datetime.now() + timedelta(days=9)).strftime('%B %d, %Y')}

Note: This is a simulated booking for demonstration purposes only."""
        
    elif "flight" in query:
        flights = [
            {"airline": "Air India", "flight": "AI-203", "price": "$350", "duration": "2h 30m"},
            {"airline": "IndiGo", "flight": "6E-456", "price": "$280", "duration": "2h 15m"},
            {"airline": "Vistara", "flight": "UK-789", "price": "$420", "duration": "2h 45m"}
        ]
        
        selected = random.choice(flights)
        booking_id = f"FLT-{random.randint(10000, 99999)}"
        
        result = f"""Flight Booking Simulated Successfully!

Booking Details:
- Booking ID: {booking_id}
- Airline: {selected['airline']}
- Flight: {selected['flight']}
- Price: {selected['price']}
- Duration: {selected['duration']}
- Departure: {(datetime.now() + timedelta(days=7)).strftime('%B %d, %Y at 10:30 AM')}
- Arrival: {(datetime.now() + timedelta(days=7)).strftime('%B %d, %Y at 1:15 PM')}

Note: This is a simulated booking for demonstration purposes only."""
    
    else:
        result = "I can help you book hotels or flights. Please specify:\n- 'Book a hotel in [city]'\n- 'Book a flight to [destination]'"
    
    return {"agent_result": result}


def weather_agent(state: AgentState) -> AgentState:
    """Get weather using Tavily search"""
    query = state["messages"][-1].content
    
    extract_prompt = f"Extract the city name from this query: '{query}'. Reply with ONLY the city name."
    city_response = llm.invoke([HumanMessage(content=extract_prompt)])
    city = city_response.content.strip()
    
    try:
        search_query = f"current weather in {city} today temperature humidity conditions"
        
        response = tavily.search(
            query=search_query,
            search_depth="basic",
            include_answer=True,
            max_results=5
        )
        
        if response.get("answer"):
            result = f"""Current Weather for {city}:

{response['answer']}

Source: Tavily AI Search"""
        elif response.get("results"):
            weather_info = "\n".join([
                f"- {r['content'][:300]}" for r in response['results'][:3]
            ])
            
            format_prompt = f"""Extract current weather information for {city} from this data:

{weather_info}

Provide a clean summary with:
- Current temperature
- Weather condition
- Humidity (if available)
- Any other relevant details

Be concise and factual."""
            
            formatted = llm.invoke([HumanMessage(content=format_prompt)])
            
            result = f"""Current Weather for {city}:

{formatted.content}

Source: Tavily AI Search"""
        else:
            result = f"Could not find weather information for {city}."
    
    except Exception as e:
        weather_conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Rainy"]
        temp = random.randint(25, 35)
        condition = random.choice(weather_conditions)
        
        result = f"""Weather for {city}:

- Temperature: {temp}C
- Condition: {condition}
- Humidity: {random.randint(60, 85)}%

Note: Using simulated data (Search unavailable: {str(e)})"""
    
    return {"agent_result": result}


def math_agent(state: AgentState) -> AgentState:
    """Handle mathematical calculations using eval"""
    query = state["messages"][-1].content

    # Use LLM to extract mathematical expression
    extract_prompt = f"""Extract the mathematical expression from this query and convert it to valid Python code.
Only use: +, -, *, /, **, (), and numbers. You can also use math functions like sqrt, sin, cos, log, etc.

Examples:
- "What is 15% of 250" -> "0.15 * 250"
- "Calculate 2 to the power of 8" -> "2 ** 8"
- "What's 45 plus 67 times 3" -> "45 + 67 * 3"
- "Square root of 144" -> "sqrt(144)"

Query: {query}

Return ONLY the Python expression, nothing else."""

    expression_response = llm.invoke([HumanMessage(content=extract_prompt)])
    expression = expression_response.content.strip().replace("```python", "").replace("```", "").strip()

    try:
        # Safe eval with restricted namespace (math functions allowed)
        import math
        safe_dict = {
            "__builtins__": {},
            "abs": abs, "round": round, "min": min, "max": max,
            "pow": pow, "sum": sum,
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "log": math.log, "log10": math.log10,
            "exp": math.exp, "pi": math.pi, "e": math.e,
            "floor": math.floor, "ceil": math.ceil
        }

        result = eval(expression, safe_dict, {})

        answer = f"""Mathematical Solution:

Expression: {expression}
Result: {result}

Calculated using Python eval()"""

    except Exception as e:
        # Fallback to LLM if eval fails
        fallback_prompt = f"""Solve this mathematical problem and show your work:

{query}

Provide a clear, step-by-step solution."""

        solution = llm.invoke([HumanMessage(content=fallback_prompt)])
        answer = f"""Mathematical Solution:

{solution.content}

Note: Used LLM solver (eval failed: {str(e)})"""

    return {"agent_result": answer}

def online_lookup_agent(state: AgentState) -> AgentState:
    """Handle general queries with RAG + Tavily search"""
    query = state["messages"][-1].content
    
    # First try RAG
    rag_results = vectorstore.similarity_search(query, k=3)
    rag_context = "\n\n".join([doc.page_content for doc in rag_results])
    
    if len(rag_context.strip()) > 50:
        judge_prompt = f"""Query: {query}
Context: {rag_context[:300]}

Can this context fully and accurately answer the query? Reply YES or NO."""
        
        judgment = llm.invoke([HumanMessage(content=judge_prompt)])
        
        if "YES" in judgment.content.upper():
            return {"agent_result": f"From knowledge base:\n\n{rag_context}"}
    
    # Use Tavily for web search
    try:
        response = tavily.search(
            query=query,
            search_depth="advanced",
            include_answer=True,
            max_results=5
        )
        
        if response.get("answer"):
            sources = "\n".join([
                f"- {r.get('title', 'Source')}: {r['url']}" 
                for r in response.get('results', [])[:3]
            ])
            result = f"{response['answer']}"
            if sources:
                result += f"\n\nSources:\n{sources}"
            return {"agent_result": result}
            
        elif response.get("results"):
            web_context = "\n\n".join([
                f"{r.get('title', 'Result')}:\n{r['content']}" 
                for r in response['results'][:3]
            ])
            sources = "\n".join([
                f"- {r['url']}" for r in response['results'][:3]
            ])
            return {"agent_result": f"{web_context}\n\nSources:\n{sources}"}
            
    except Exception as e:
        return {"agent_result": f"Search failed: {str(e)}"}
    
    return {"agent_result": "No information found."}

# ========== ANSWER NODE (with conversation history) ==========
def answer_node(state: AgentState) -> AgentState:
    """Generate final answer using agent results and conversation history"""
    agent_result = state.get("agent_result", "")
    messages = state.get("messages", [])
    current_query = messages[-1].content if messages else ""
    
    # Build conversation history for context
    conversation_history = ""
    if len(messages) > 1:
        history_messages = messages[-10:]  # Last 10 messages for context
        for msg in history_messages[:-1]:  # Exclude current message
            role = "User" if isinstance(msg, HumanMessage) else "Assistant"
            conversation_history += f"{role}: {msg.content}\n"
    
    prompt = f"""You are a helpful assistant. Use the conversation history and information provided to give a relevant response.

Conversation History:
{conversation_history if conversation_history else "No previous conversation."}

Current User Query: {current_query}

Information from tools: {agent_result}

Instructions:
- If the user asks about something mentioned in the conversation history, use that context.
- If the user asks personal questions (like their name), check if they mentioned it in the conversation history.
- Provide a natural, conversational response.
- Include source links if available from the tool information."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": [response]}

# ========== SUPERVISOR ROUTING NODE ==========
def supervisor_routing(state: AgentState) -> AgentState:
    """Passthrough node to enable supervisor conditional edges"""
    return state

# ========== BUILD GRAPH ==========
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("llm_node", llm_node)
graph.add_node("supervisor_routing", supervisor_routing)
graph.add_node("booking", booking_agent)
graph.add_node("weather", weather_agent)
graph.add_node("math", math_agent)
graph.add_node("online_lookup", online_lookup_agent)
graph.add_node("answer", answer_node)

graph.add_edge(START, "llm_node")

graph.add_conditional_edges(
    "llm_node",
    route_after_llm,
    {
        "blocked": END,
        "continue": "supervisor_routing"
    }
)

graph.add_conditional_edges(
    "supervisor_routing",
    supervisor,
    {
        "booking": "booking",
        "weather": "weather",
        "math": "math",
        "online_lookup": "online_lookup"
    }
)

graph.add_edge("booking", "answer")
graph.add_edge("weather", "answer")
graph.add_edge("math", "answer")
graph.add_edge("online_lookup", "answer")

graph.add_edge("answer", END)

# Initialize SQLite persistence
conn = sqlite3.connect("./checkpoints.db", check_same_thread=False)
memory = SqliteSaver(conn)

# Compile graph with memory
app = graph.compile(checkpointer=memory)

if __name__ == "__main__":
    print("=" * 60)
    print("Multi-Agent Chatbot (Powered by Tavily)")
    print("=" * 60)
    print("I can help with:")
    print("  - Bookings (hotels & flights)")
    print("  - Weather forecasts")
    print("  - Math calculations")
    print("  - General knowledge & web search")
    print("\nType 'quit' to exit")
    print("=" * 60)
    
    # Hardcoded thread_id for persistent memory
    config = {"configurable": {"thread_id": "1"}}
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        try:
            result = app.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config
            )
            
            print(f"\nAgent: {result['messages'][-1].content}")
            
        except Exception as e:
            print(f"\nError: {str(e)}")
from IPython.display import Image, display

# Save graph as PNG
graph_png = app.get_graph().draw_mermaid_png()
with open("graph_diagram.png", "wb") as f:
    f.write(graph_png)