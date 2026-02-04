import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# Import the compiled app from react agent
from react_agent import app as react_app

# Page configuration
st.set_page_config(
    page_title="ReAct Agent Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .tool-badge {
        background-color: #4CAF50;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
        font-size: 12px;
        margin-right: 5px;
    }
    .stChatMessage {
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.title("🤖 ReAct Agent")
    st.markdown("---")

    # Login / Thread ID for memory persistence
    st.subheader("User Session")
    user_id = st.text_input("Enter User ID (for memory persistence)", value="user1")

    st.markdown("---")

    # Available tools info
    st.subheader("Available Tools")
    st.markdown("""
    - 🌤️ **Weather** - Get current weather for any city
    - 🧮 **Math** - Calculate mathematical expressions
    - 🏨 **Hotel Booking** - Book hotels (simulated)
    - ✈️ **Flight Booking** - Book flights (simulated)
    - 🔍 **Web Search** - Search the web for information
    """)

    st.markdown("---")

    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown("---")
    st.caption("Powered by LangGraph + OpenAI")

# Main chat interface
st.title("💬 Multi-Agent Chatbot")
st.caption("Ask me about weather, math, bookings, or anything else!")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("tools_used"):
            tools_html = " ".join([f'<span class="tool-badge">🔧 {tool}</span>' for tool in message["tools_used"]])
            st.markdown(f"Tools used: {tools_html}", unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Configure with user's thread_id
                config = {"configurable": {"thread_id": user_id}}

                tools_used = []
                final_response = ""

                # Stream the response
                for event in react_app.stream(
                    {"messages": [HumanMessage(content=prompt)]},
                    config=config,
                    stream_mode="values"
                ):
                    if event.get("messages"):
                        last_msg = event["messages"][-1]

                        # Track tool calls
                        if hasattr(last_msg, 'tool_calls') and last_msg.tool_calls:
                            tools_used.extend([tc['name'] for tc in last_msg.tool_calls])

                        # Get final AI response
                        if isinstance(last_msg, AIMessage) and last_msg.content:
                            if not (hasattr(last_msg, 'tool_calls') and last_msg.tool_calls):
                                final_response = last_msg.content

                # Display the response
                st.markdown(final_response)

                # Show tools used
                if tools_used:
                    unique_tools = list(set(tools_used))
                    tools_html = " ".join([f'<span class="tool-badge">🔧 {tool}</span>' for tool in unique_tools])
                    st.markdown(f"Tools used: {tools_html}", unsafe_allow_html=True)

                # Add assistant message to chat history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_response,
                    "tools_used": list(set(tools_used)) if tools_used else None
                })

            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })
