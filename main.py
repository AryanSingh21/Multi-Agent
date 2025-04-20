# app.py
import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from typing import Annotated, List, Dict, Any, Optional, Union
from typing_extensions import TypedDict
from PIL import Image
import base64
import io
import uuid
from groq import Groq
# Import LangGraph components
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Set environment variables
# os.environ["GROQ_API_KEY"] = st.secrets.get("GROQ_API_KEY", "")

# Initialize the LLM
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.2 ,api_key="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")
groq_client = Groq(api_key="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")

# Define the state schema with proper annotations
class AgentState(TypedDict):
    # Messages will use add_messages to append rather than overwrite
    messages: Annotated[List[Union[HumanMessage, AIMessage, SystemMessage]], add_messages]
    # Other state fields will be overwritten when updated
    current_agent: str
    image_data: Optional[str]
    needs_clarification: bool
    analysis_results: Optional[Dict]

# Define agent functions
def router(state: AgentState) -> Dict:
    """Routes the query to the appropriate specialized agent."""
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    has_image = state.get("image_data") is not None
    print("kfjubsafudbisubdcuisbcidscbidh", has_image)
    # Only process if the last message is from the user
    if not isinstance(last_message, HumanMessage):
        return {}
    
    # Extract content from the message
    user_content = last_message.content
    
    # Create routing prompt
    prompt = f"""Determine which specialized real estate agent should handle this query:
    
    User Query: {user_content}
    Has Image: {'Yes' if has_image else 'No'}
    
    Choose ONE of these exact options:
    - "ISSUE_AGENT" (for problems, maintenance, or if an image is present)
    - "TENANCY_AGENT" (for rental agreements, tenant rights, contracts, landlord responsibilities)
    - "CLARIFY" (if you need more information to route properly)
    
    Respond with only one of these exact strings.
    """
    
    response = llm.invoke(prompt)

    print(response)
    decision = response.content.strip()
    
    # Return the routing decision
    if "ISSUE_AGENT" in decision:
        return {"current_agent": "issue_agent", "needs_clarification": False}
    elif "TENANCY_AGENT" in decision:
        return {"current_agent": "tenancy_agent", "needs_clarification": False}
    else:
        return {"current_agent": "clarify_agent", "needs_clarification": True}

def issue_agent(state: AgentState) -> Dict:
    """Handles property issues and image analysis using vision capabilities."""
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    image_data = state.get("image_data")
    
    if not isinstance(last_message, HumanMessage):
        return {}
    
    user_content = last_message.content
    
    # If we have an image, use the vision model
    if image_data:
        try:
            # Prepare message with both text and image for vision analysis
            vision_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"Analyze this property image and identify any issues, damage, or maintenance problems. The user said: '{user_content}'"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}",
                            },
                        },
                    ],
                }
            ]
            
            # Call the vision-enabled model
            vision_response = groq_client.chat.completions.create(
                messages=vision_messages,
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                temperature=0.2
            )
            
            image_analysis = vision_response.choices[0].message.content
            
            # Now create a comprehensive response with the analysis
            prompt = f"""You are a Property Issue Detection & Troubleshooting Agent for a real estate assistant.
            
            User query: {user_content}
            
            Image analysis: {image_analysis}
            
            Provide a helpful response that:
            1. References the image the user uploaded
            2. Explains what issues were detected in the image
            3. Suggests practical solutions or professionals to contact
            4. Asks relevant follow-up questions if needed
            
            Respond in a professional, helpful tone.
            """
            
            response = llm.invoke(prompt)
            
            return {
                "messages": [AIMessage(content=response.content)],
                "analysis_results": {"issue_detected": True, "image_processed": True, "analysis": image_analysis}
            }
            
        except Exception as e:
            # Fallback if vision analysis fails
            st.error(f"Error analyzing image: {str(e)}")
            prompt = f"""You are a Property Issue Detection & Troubleshooting Agent for a real estate assistant.
            
            User query: {user_content}
            
            Note: The user uploaded an image but there was an error analyzing it. Please focus on their text description.
            
            Provide a helpful response that:
            1. Acknowledges the property issue from their description
            2. Explains what might be causing it
            3. Suggests practical solutions or professionals to contact
            4. Asks for more details about what was shown in the image
            
            Respond in a professional, helpful tone.
            """
            
            response = llm.invoke(prompt)
            
            return {
                "messages": [AIMessage(content=response.content)],
                "analysis_results": {"issue_detected": True, "image_processed": False, "error": str(e)}
            }
    else:
        # No image, use standard text processing
        prompt = f"""You are a Property Issue Detection & Troubleshooting Agent for a real estate assistant.
        
        User query: {user_content}
        
        Provide a helpful response that:
        1. Acknowledges the property issue
        2. Explains what might be causing it
        3. Suggests practical solutions or professionals to contact
        4. Asks relevant follow-up questions if needed
        
        Respond in a professional, helpful tone.
        """
        
        response = llm.invoke(prompt)
        
        return {
            "messages": [AIMessage(content=response.content)],
            "analysis_results": {"issue_detected": True, "image_processed": False}
        }


def tenancy_agent(state: AgentState) -> Dict:
    """Handles tenancy-related queries."""
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    
    if not isinstance(last_message, HumanMessage):
        return {}
    
    user_content = last_message.content
    
    prompt = f"""You are a Tenancy FAQ Agent specializing in rental agreements, tenant rights, and landlord responsibilities.
    
    User query: {user_content}
    
    Provide an informative response that:
    1. Addresses the user's specific tenancy question
    2. Explains relevant legal concepts in simple terms
    3. Offers practical advice while noting this isn't legal counsel
    4. Asks for location details if necessary for a more specific answer
    
    Respond in a professional, helpful tone.
    """
    
    response = llm.invoke(prompt)
    
    # Return response message to be added to state
    return {
        "messages": [AIMessage(content=response.content)]
    }

def clarify_agent(state: AgentState) -> Dict:
    """Asks clarifying questions to better route the query."""
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    
    if not isinstance(last_message, HumanMessage):
        return {}
    
    user_content = last_message.content
    
    prompt = f"""You need to ask clarifying questions to better understand the user's real estate query.
    
    User's query: {user_content}
    
    Ask 1-2 specific questions to determine if they need:
    1. Help with a property issue/maintenance problem
    2. Information about tenancy, rental agreements, or rights
    
    Keep your response short and focused on getting the necessary information.
    """
    
    response = llm.invoke(prompt)
    
    # Return response message to be added to state
    return {
        "messages": [AIMessage(content=response.content)]
    }

# Define a routing function that returns the next node name
def route_to_agent(state: AgentState) -> str:
    """Determines which agent to route to based on the state"""
    return state["current_agent"]

# Build the LangGraph
def build_graph():
    """Builds the agent graph using LangGraph."""
    # Create state graph
    graph_builder = StateGraph(AgentState)
    
    # Add nodes
    graph_builder.add_node("router", router)
    graph_builder.add_node("issue_agent", issue_agent)
    graph_builder.add_node("tenancy_agent", tenancy_agent)
    graph_builder.add_node("clarify_agent", clarify_agent)
    
    # Add edges
    graph_builder.add_edge(START, "router")
    
    # Add conditional edges using the proper pattern
    graph_builder.add_conditional_edges(
        "router",  # from node
        route_to_agent,  # function that returns the name of the next node
        {
            "issue_agent": "issue_agent",
            "tenancy_agent": "tenancy_agent",
            "clarify_agent": "clarify_agent"
        }
    )
    
    # Connect to END
    graph_builder.add_edge("issue_agent", END)
    graph_builder.add_edge("tenancy_agent", END)
    graph_builder.add_edge("clarify_agent", END)
    
    # Compile the graph
    return graph_builder.compile()

# Initialize graph
graph = build_graph()

# Streamlit UI
st.title("Real Estate Assistant")
st.markdown("### Your multi-agent property and tenancy helper")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = str(uuid.uuid4())

# Display chat messages
for message in st.session_state.messages:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    content = message.content
    
    with st.chat_message(role):
        st.markdown(content)
        if hasattr(message, "image") and message.image:
            st.image(message.image)

# Handle file upload for image
uploaded_file = st.file_uploader("Upload an image of your property issue", type=["jpg", "jpeg", "png"])
image_data = None
image_display = None

if uploaded_file:
    image = Image.open(uploaded_file)
    image_display = image
    
    # Convert image to base64 for potential API usage
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="PNG")
    image_data = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
    
    st.image(image, caption="Uploaded image", use_column_width=True)

# Get user input
user_input = st.chat_input("Ask about property issues or tenancy questions...")

if user_input:
    # Create a proper HumanMessage
    user_message = HumanMessage(content=user_input)
    
    # Add image attribute if available
    if image_display:
        setattr(user_message, "image", image_display)
    
    # Add to session state
    st.session_state.messages.append(user_message)
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
        if image_display:
            st.image(image_display)
    
    # Create initial state for graph
    initial_state = {
        "messages": [user_message],
        "current_agent": "router",  # Default to router first
        "image_data": image_data,
        "needs_clarification": False,
        "analysis_results": None
    }
    
    # Process through the graph
    with st.spinner("Thinking..."):
        result = None
        # Stream results for better UX (shows thinking progress)
        for event in graph.stream(initial_state):
            result = event
    print(result)
    responding_agent = next(iter(result.keys()))
    # Then extract the messages from that agent's response
    if "messages" in result[responding_agent]:
        assistant_responses = [msg for msg in result[responding_agent]["messages"] if isinstance(msg, AIMessage)]
    else:
        assistant_responses = []

    
    if assistant_responses:
        assistant_response = assistant_responses[-1]
        # Add to session state
        st.session_state.messages.append(assistant_response)
        
        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(assistant_response.content)
    else:
        st.error("No response generated. Please try again.")