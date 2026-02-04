from dotenv import load_dotenv
load_dotenv()

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
#from duckduckgo_search import DDGS  # Add this import

@tool
def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It's always sunny in {city}!"

@tool
def search_duckduckgo(query: str) -> str:
    """Search DuckDuckGo for information about a query.
    
    Args:
        query: The search query string
        
    Returns:
        A summary of the top search results
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
            
        if not results:
            return "No results found."
        
        # Format results
        output = []
        for i, result in enumerate(results, 1):
            output.append(f"{i}. {result['title']}")
            output.append(f"   {result['body']}")
            output.append(f"   URL: {result['href']}\n")
        
        return "\n".join(output)
    
    except Exception as e:
        return f"Search failed: {str(e)}"

llm = ChatOpenAI(model="gpt-4o-mini")

# Create agent with both tools
agent = create_react_agent(
    llm, 
    tools=[get_weather, search_duckduckgo]
)

# Test with search query
result = agent.invoke({
    "messages": [
        ("system", "You are a helpful assistant."),
        ("user", "What is 2*3+4")
    ]
})

print(result["messages"][-1].content)