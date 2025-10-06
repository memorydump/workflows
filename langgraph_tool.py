"""
LangGraph Agent with MCP Tool Integration

This example shows how to build a LangGraph agent that:
1. Calls a tool via MCP (Model Context Protocol)
2. Saves the tool response in the state
3. Returns the result
"""

from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_anthropic import ChatAnthropic
import operator

# Define the state structure
class AgentState(TypedDict):
    """State schema for the agent"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    tool_response: str  # Store the tool response here
    final_output: str   # Final output to return


# Example MCP tool definition
# In practice, you'd import this from your MCP server
from langchain_core.tools import tool

@tool
def example_mcp_tool(query: str) -> str:
    """
    Example MCP tool that processes a query.
    Replace this with your actual MCP tool.
    
    Args:
        query: The input query to process
        
    Returns:
        The processed result
    """
    # This would be your actual MCP tool call
    return f"MCP Tool Response: Processed '{query}'"


# Define the agent node
def agent_node(state: AgentState):
    """
    Agent node that decides whether to call a tool or finish.
    """
    messages = state["messages"]
    
    # Initialize the model with tools
    model = ChatAnthropic(model="claude-sonnet-4-5-20250929", temperature=0)
    model_with_tools = model.bind_tools([example_mcp_tool])
    
    # Get the model's response
    response = model_with_tools.invoke(messages)
    
    # Return updated state
    return {"messages": [response]}


# Define the tool execution node
def tool_execution_node(state: AgentState):
    """
    Execute the tool and save the response in state.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # Execute the tool
    tool_node = ToolNode([example_mcp_tool])
    tool_result = tool_node.invoke({"messages": messages})
    
    # Extract the tool response content
    tool_response_content = tool_result["messages"][-1].content
    
    # Save to state
    return {
        "messages": tool_result["messages"],
        "tool_response": tool_response_content
    }


# Define the finalize node
def finalize_node(state: AgentState):
    """
    Finalize the output with the tool response.
    """
    tool_response = state.get("tool_response", "")
    
    # Create final output
    final_output = f"Agent completed. Tool response: {tool_response}"
    
    return {"final_output": final_output}


# Router function to decide next step
def should_continue(state: AgentState):
    """
    Determine whether to continue to tools or end.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # If there are tool calls, continue to tool execution
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    # Otherwise, finalize
    return "finalize"


# Build the graph
def create_agent_graph():
    """
    Create and compile the LangGraph agent.
    """
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_execution_node)
    workflow.add_node("finalize", finalize_node)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "finalize": "finalize"
        }
    )
    
    # After tools, go back to agent
    workflow.add_edge("tools", "agent")
    
    # After finalize, end
    workflow.add_edge("finalize", END)
    
    return workflow.compile()


# Usage example
if __name__ == "__main__":
    # Create the agent
    agent = create_agent_graph()
    
    # Run the agent
    initial_state = {
        "messages": [HumanMessage(content="Please use the MCP tool to process 'hello world'")],
        "tool_response": "",
        "final_output": ""
    }
    
    # Execute
    result = agent.invoke(initial_state)
    
    # Access the results
    print("Final Output:", result["final_output"])
    print("\nTool Response:", result["tool_response"])
    print("\nAll Messages:")
    for msg in result["messages"]:
        print(f"  {type(msg).__name__}: {msg.content}")
