import os
from typing import TypedDict, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class LangGraphAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.graph = self._create_graph()
        
    def _create_graph(self):
        # Define the graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("agent", self._call_model)
        
        # Set entry point
        workflow.set_entry_point("agent")
        
        # Add edge from agent to end
        workflow.add_edge("agent", END)
        
        # Compile the graph
        return workflow.compile()
    
    def _call_model(self, state: AgentState):
        messages = state["messages"]
        response = self.llm.invoke(messages)
        return {"messages": [response]}
    
    async def process_message(self, message: str) -> str:
        """Process a message through the LangGraph agent"""
        try:
            # Create initial state
            initial_state = {"messages": [HumanMessage(content=message)]}
            
            # Run the graph
            result = self.graph.invoke(initial_state)
            
            # Extract the response
            last_message = result["messages"][-1]
            return last_message.content
            
        except Exception as e:
            return f"Error processing message: {str(e)}"
