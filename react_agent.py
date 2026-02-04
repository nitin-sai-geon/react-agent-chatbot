from typing import Annotated, TypedDict, Literal
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.tools import tool
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from tavily import TavilyClient
import sqlite3
import random
import os
import math
from datetime import datetime, timedelta

load_dotenv()

# Initialize Tavily client
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
vectorstore = Chroma(
    collection_name="rag_knowledge_base",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    is_blocked: bool

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4)

# ========== TOOLS USING @tool DECORATOR ==========

@tool
def get_weather(city: str) -> str:
    """Get current weather for a specified city using Tavily API.

    Args:
        city: The name of the city to get weather for

    Returns:
        Weather information including temperature, conditions, and humidity
    """
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

Provide a clean summary with temperature, weather condition, humidity (if available)."""

            formatted = llm.invoke([HumanMessage(content=format_prompt)])

            result = f"""Current Weather for {city}:

{formatted.content}

Source: Tavily AI Search"""
        else:
            result = f"Could not find weather information for {city}."

    except Exception as e:
        # Fallback to simulated data
        weather_conditions = ["Sunny", "Partly Cloudy", "Cloudy", "Rainy"]
        temp = random.randint(25, 35)
        condition = random.choice(weather_conditions)

        result = f"""Weather for {city}:

- Temperature: {temp}°C
- Condition: {condition}
- Humidity: {random.randint(60, 85)}%

Note: Using simulated data (Search unavailable: {str(e)})"""

    return result


@tool
def calculate_math(expression: str) -> str:
    """Calculate mathematical expressions. Pass the expression as a string.

    Args:
        expression: Math expression in natural language (e.g., "15% of 250", "2 to the power of 8")

    Returns:
        The calculated result
    """
    # Use LLM to extract mathematical expression
    extract_prompt = f"""Extract the mathematical expression from this query and convert it to valid Python code.
Only use: +, -, *, /, **, (), and numbers. You can also use math functions like sqrt, sin, cos, log, etc.

Examples:
- "15% of 250" -> "0.15 * 250"
- "2 to the power of 8" -> "2 ** 8"
- "45 plus 67 times 3" -> "45 + 67 * 3"
- "square root of 144" -> "sqrt(144)"

Query: {expression}

Return ONLY the Python expression, nothing else."""

    expression_response = llm.invoke([HumanMessage(content=extract_prompt)])
    python_expr = expression_response.content.strip().replace("```python", "").replace("```", "").strip()

    try:
        # Safe eval with restricted namespace
        safe_dict = {
            "__builtins__": {},
            "abs": abs, "round": round, "min": min, "max": max,
            "pow": pow, "sum": sum,
            "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos,
            "tan": math.tan, "log": math.log, "log10": math.log10,
            "exp": math.exp, "pi": math.pi, "e": math.e,
            "floor": math.floor, "ceil": math.ceil
        }

        result = eval(python_expr, safe_dict, {})
        return f"Expression: {python_expr}\nResult: {result}"

    except Exception as e:
        return f"Could not calculate: {str(e)}"


@tool
def book_hotel(location: str) -> str:
    """Simulate booking a hotel in a specified location.

    Args:
        location: City or area where the hotel should be booked

    Returns:
        Simulated hotel booking confirmation with details
    """
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
- Location: {selected['location']} ({location})
- Price: {selected['price']}
- Rating: {selected['rating']}
- Check-in: {(datetime.now() + timedelta(days=7)).strftime('%B %d, %Y')}
- Check-out: {(datetime.now() + timedelta(days=9)).strftime('%B %d, %Y')}

Note: This is a simulated booking for demonstration purposes only."""

    return result


@tool
def book_flight(destination: str) -> str:
    """Simulate booking a flight to a specified destination.

    Args:
        destination: Destination city for the flight

    Returns:
        Simulated flight booking confirmation with details
    """
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
- Destination: {destination}
- Airline: {selected['airline']}
- Flight: {selected['flight']}
- Price: {selected['price']}
- Duration: {selected['duration']}
- Departure: {(datetime.now() + timedelta(days=7)).strftime('%B %d, %Y at 10:30 AM')}
- Arrival: {(datetime.now() + timedelta(days=7)).strftime('%B %d, %Y at 1:15 PM')}

Note: This is a simulated booking for demonstration purposes only."""

    return result


@tool
def web_search(query: str) -> str:
    """Search the web for current information using Tavily API and RAG.

    Args:
        query: The search query or question

    Returns:
        Information from web search or knowledge base with sources
    """
    # First try RAG
    rag_results = vectorstore.similarity_search(query, k=3)
    rag_context = "\n\n".join([doc.page_content for doc in rag_results])

    if len(rag_context.strip()) > 50:
        judge_prompt = f"""Query: {query}
Context: {rag_context[:300]}

Can this context fully and accurately answer the query? Reply YES or NO."""

        judgment = llm.invoke([HumanMessage(content=judge_prompt)])

        if "YES" in judgment.content.upper():
            return f"From knowledge base:\n\n{rag_context}"

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
            return result

        elif response.get("results"):
            web_context = "\n\n".join([
                f"{r.get('title', 'Result')}:\n{r['content']}"
                for r in response['results'][:3]
            ])
            sources = "\n".join([
                f"- {r['url']}" for r in response['results'][:3]
            ])
            return f"{web_context}\n\nSources:\n{sources}"

    except Exception as e:
        return f"Search failed: {str(e)}"

    return "No information found."


# ========== BIND TOOLS TO MODEL ==========
tools = [get_weather, calculate_math, book_hotel, book_flight, web_search]
llm_with_tools = llm.bind_tools(tools)

# Create tool node for automatic tool execution
tool_node = ToolNode(tools)


# ========== GRAPH NODES ==========

def prompt_protection_node(state: AgentState) -> AgentState:
    """Check for prompt injection attempts"""
    user_message = state["messages"][-1].content.lower()

    blocked_keywords = [
        "system prompt", "ignore previous", "ignore instructions",
        "you are now", "forget everything", "system:", "assistant:",
        "reveal your prompt", "show me your instructions",
        "disregard", "override", "bypass", "jailbreak"
    ]

    if any(keyword in user_message for keyword in blocked_keywords):
        return {
            "messages": [AIMessage(content="I cannot process that request.")],
            "is_blocked": True
        }

    return {"is_blocked": False}


def agent_node(state: AgentState) -> AgentState:
    """ReAct agent with tools bound to model"""
    messages = state["messages"]

    # Add system message for tool usage
    system_message = """You are a helpful assistant with access to tools. Use them when needed.

Available tools:
- get_weather: Get current weather for a city
- calculate_math: Calculate mathematical expressions
- book_hotel: Book hotels (simulated)
- book_flight: Book flights (simulated)
- web_search: Search the web for information

When you need information, call the appropriate tool. After getting tool results, provide a helpful response to the user."""

    messages_with_system = [HumanMessage(content=system_message)] + messages

    # Invoke LLM with tools bound
    response = llm_with_tools.invoke(messages_with_system)

    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Determine if we should call tools or end"""
    messages = state["messages"]
    last_message = messages[-1]

    # If the LLM makes a tool call, route to tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"

    # Otherwise, end
    return "end"


def check_blocked(state: AgentState) -> Literal["blocked", "continue"]:
    """Check if prompt was blocked"""
    if state.get("is_blocked", False):
        return "blocked"
    return "continue"


# ========== BUILD GRAPH ==========
graph = StateGraph(AgentState)

# Add nodes
graph.add_node("prompt_protection", prompt_protection_node)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

# Add edges
graph.add_edge(START, "prompt_protection")

graph.add_conditional_edges(
    "prompt_protection",
    check_blocked,
    {
        "blocked": END,
        "continue": "agent"
    }
)

graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)

# After tools execute, go back to agent
graph.add_edge("tools", "agent")

# Initialize SQLite persistence
conn = sqlite3.connect("./checkpoints_react.db", check_same_thread=False)
memory = SqliteSaver(conn)

# Compile graph with memory
app = graph.compile(checkpointer=memory)

if __name__ == "__main__":
    print("=" * 70)
    print("ReAct Agent with Tools Bound to Model")
    print("=" * 70)
    print("I can help with:")
    print("  - Weather forecasts (tool: get_weather)")
    print("  - Math calculations (tool: calculate_math)")
    print("  - Hotel bookings (tool: book_hotel)")
    print("  - Flight bookings (tool: book_flight)")
    print("  - Web searches (tool: web_search)")
    print("\nThe LLM decides which tools to call automatically!")
    print("\nType 'quit' to exit")
    print("=" * 70)

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
            # Stream the response to show tool calls
            print("\nAgent: ", end="", flush=True)

            for event in app.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="values"
            ):
                # Get the last message
                if event.get("messages"):
                    last_msg = event["messages"][-1]

                    # Show tool calls if present
                    if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                        print(f"\n[Calling tools: {[tc['name'] for tc in last_msg.tool_calls]}]", flush=True)

                    # Show final AI response
                    if isinstance(last_msg, AIMessage) and last_msg.content:
                        if not hasattr(last_msg, 'tool_calls') or not last_msg.tool_calls:
                            print(last_msg.content, flush=True)

        except Exception as e:
            print(f"\nError: {str(e)}")

# Save graph diagram
try:
    graph_png = app.get_graph().draw_mermaid_png()
    with open("graph_diagramfinal_react.png", "wb") as f:
        f.write(graph_png)
    print("\n✅ Graph diagram saved as 'graph_diagram_react.png'")
except:
    pass
