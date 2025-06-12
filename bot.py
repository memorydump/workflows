from typing import Dict, List, Any, Optional, Literal
from typing_extensions import TypedDict
import json
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# Define the state structure
class BondCriteria(BaseModel):
    """Individual bond criteria item"""
    category: str = Field(description="Category of criteria (maturity, sector, ticker, etc.)")
    value: str = Field(description="The specific value or constraint")
    operation: Optional[str] = Field(default=None, description="Operation type (add, replace, remove)")

class PortfolioState(TypedDict):
    """State maintained throughout the conversation"""
    messages: List[BaseMessage]
    bond_universe: List[BondCriteria]
    optimization_constraints: List[Dict[str, Any]]
    current_intent: Optional[str]
    portfolio_built: bool
    portfolio_data: Optional[Dict[str, Any]]


class BondPortfolioAgent:
    def __init__(self, api_key: str, portfolio_api_endpoint: str = None):
        self.llm = ChatOpenAI(
            model="gpt-4-turbo-preview",
            api_key=api_key,
            temperature=0.1
        )
        self.portfolio_api_endpoint = portfolio_api_endpoint
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(PortfolioState)
        
        # Add nodes
        workflow.add_node("analyze_intent", self._analyze_intent)
        workflow.add_node("capture_bond_criteria", self._capture_bond_criteria)
        workflow.add_node("capture_optimization_constraints", self._capture_optimization_constraints)
        workflow.add_node("build_portfolio", self._build_portfolio)
        workflow.add_node("query_portfolio", self._query_portfolio)
        workflow.add_node("acknowledge_update", self._acknowledge_update)
        
        # Set entry point
        workflow.set_entry_point("analyze_intent")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "analyze_intent",
            self._route_based_on_intent,
            {
                "bond_criteria": "capture_bond_criteria",
                "optimization": "capture_optimization_constraints", 
                "build_portfolio": "build_portfolio",
                "query_portfolio": "query_portfolio",
                "general": "acknowledge_update"
            }
        )
        
        # Add edges from other nodes
        workflow.add_edge("capture_bond_criteria", "acknowledge_update")
        workflow.add_edge("capture_optimization_constraints", "acknowledge_update")
        workflow.add_edge("build_portfolio", "acknowledge_update")
        workflow.add_edge("query_portfolio", "acknowledge_update")
        workflow.add_edge("acknowledge_update", END)
        
        return workflow.compile()
    
    def _analyze_intent(self, state: PortfolioState) -> PortfolioState:
        """Analyze user intent from the latest message"""
        latest_message = state["messages"][-1].content
        
        intent_prompt = ChatPromptTemplate.from_template("""
        Analyze the user's message and determine their intent. Return only one of these categories:
        - "bond_criteria": User is specifying bond universe characteristics (maturity, sector, ticker, ISIN, rating, etc.)
        - "optimization": User is specifying optimization constraints (duration, yield, allocation limits, etc.)
        - "build_portfolio": User wants to build/create the portfolio
        - "query_portfolio": User is asking questions about an already built portfolio
        - "general": General conversation or unclear intent
        
        User message: {message}
        
        Current bond universe criteria: {bond_universe}
        
        Intent:""")
        
        response = self.llm.invoke(intent_prompt.format(
            message=latest_message,
            bond_universe=state.get("bond_universe", [])
        ))
        
        state["current_intent"] = response.content.strip()
        return state
    
    def _capture_bond_criteria(self, state: PortfolioState) -> PortfolioState:
        """Extract and update bond universe criteria"""
        latest_message = state["messages"][-1].content
        current_criteria = state.get("bond_universe", [])
        
        extraction_prompt = ChatPromptTemplate.from_template("""
        Extract bond universe criteria from the user's message. Return a JSON list of criteria objects.
        Each object should have: category, value, and operation (add/replace/remove).
        
        Categories include: maturity, sector, ticker, isin, rating, coupon, duration, issuer, currency, etc.
        
        User message: {message}
        Current criteria: {current_criteria}
        
        If the user is modifying existing criteria, mark operation as "replace" or "remove".
        If adding new criteria, mark as "add".
        
        Return only valid JSON:""")
        
        response = self.llm.invoke(extraction_prompt.format(
            message=latest_message,
            current_criteria=json.dumps([c.dict() if isinstance(c, BondCriteria) else c for c in current_criteria])
        ))
        
        try:
            new_criteria_data = json.loads(response.content)
            new_criteria = [BondCriteria(**item) for item in new_criteria_data]
            
            # Update the bond universe based on operations
            updated_criteria = list(current_criteria)
            
            for criterion in new_criteria:
                if criterion.operation == "replace":
                    # Remove existing criteria of the same category
                    updated_criteria = [c for c in updated_criteria if c.category != criterion.category]
                    updated_criteria.append(criterion)
                elif criterion.operation == "remove":
                    updated_criteria = [c for c in updated_criteria if not (c.category == criterion.category and c.value == criterion.value)]
                else:  # add
                    updated_criteria.append(criterion)
            
            state["bond_universe"] = updated_criteria
            
        except json.JSONDecodeError:
            # Fallback: add as general criteria
            state["bond_universe"] = current_criteria
        
        return state
    
    def _capture_optimization_constraints(self, state: PortfolioState) -> PortfolioState:
        """Extract and update optimization constraints"""
        latest_message = state["messages"][-1].content
        current_constraints = state.get("optimization_constraints", [])
        
        extraction_prompt = ChatPromptTemplate.from_template("""
        Extract optimization constraints from the user's message. Return a JSON list of constraint objects.
        Each object should have: type, value, operator (min/max/equal/range), and description.
        
        Constraint types include: duration, yield, allocation, risk, concentration, etc.
        
        User message: {message}
        Current constraints: {current_constraints}
        
        Return only valid JSON:""")
        
        response = self.llm.invoke(extraction_prompt.format(
            message=latest_message,
            current_constraints=json.dumps(current_constraints)
        ))
        
        try:
            new_constraints = json.loads(response.content)
            state["optimization_constraints"] = current_constraints + new_constraints
        except json.JSONDecodeError:
            pass
        
        return state
    
    def _build_portfolio(self, state: PortfolioState) -> PortfolioState:
        """Build the portfolio using the API"""
        bond_universe = state.get("bond_universe", [])
        optimization_constraints = state.get("optimization_constraints", [])
        
        # Prepare API payload
        api_payload = {
            "bond_universe": [c.dict() if isinstance(c, BondCriteria) else c for c in bond_universe],
            "optimization_constraints": optimization_constraints
        }
        
        # Simulate API call (replace with actual API call)
        if self.portfolio_api_endpoint:
            # portfolio_data = self._call_portfolio_api(api_payload)
            pass
        
        # Mock portfolio data for demonstration
        portfolio_data = {
            "status": "success",
            "portfolio_id": "PORT_001",
            "total_bonds": 15,
            "total_value": 1000000,
            "average_maturity": 5.2,
            "average_yield": 3.8,
            "duration": 4.7,
            "bonds": [
                {"ticker": "US10Y", "weight": 0.2, "maturity": "2034-06-15"},
                {"ticker": "CORP_AAA", "weight": 0.15, "maturity": "2029-12-01"},
                # ... more bonds
            ]
        }
        
        state["portfolio_built"] = True
        state["portfolio_data"] = portfolio_data
        
        return state
    
    def _query_portfolio(self, state: PortfolioState) -> PortfolioState:
        """Answer questions about the built portfolio"""
        if not state.get("portfolio_built", False):
            return state
        
        latest_message = state["messages"][-1].content
        portfolio_data = state.get("portfolio_data", {})
        
        query_prompt = ChatPromptTemplate.from_template("""
        Answer the user's question about the portfolio based on the portfolio data.
        Be precise and factual.
        
        User question: {question}
        Portfolio data: {portfolio_data}
        
        Answer:""")
        
        response = self.llm.invoke(query_prompt.format(
            question=latest_message,
            portfolio_data=json.dumps(portfolio_data, indent=2)
        ))
        
        # This will be handled in acknowledge_update
        state["_query_response"] = response.content
        
        return state
    
    def _acknowledge_update(self, state: PortfolioState) -> PortfolioState:
        """Generate acknowledgment message"""
        intent = state.get("current_intent", "general")
        
        if intent == "bond_criteria":
            bond_universe = state.get("bond_universe", [])
            latest_criteria = bond_universe[-1] if bond_universe else None
            
            if latest_criteria:
                acknowledgment = f"✓ Captured bond criteria: {latest_criteria.category} = {latest_criteria.value}"
                if len(bond_universe) > 1:
                    acknowledgment += f"\n\nCurrent bond universe ({len(bond_universe)} criteria):"
                    for i, criteria in enumerate(bond_universe, 1):
                        acknowledgment += f"\n{i}. {criteria.category}: {criteria.value}"
            else:
                acknowledgment = "Bond criteria captured."
                
        elif intent == "optimization":
            constraints = state.get("optimization_constraints", [])
            acknowledgment = f"✓ Optimization constraint added. Total constraints: {len(constraints)}"
            
        elif intent == "build_portfolio":
            if state.get("portfolio_built", False):
                portfolio_data = state.get("portfolio_data", {})
                acknowledgment = f"""✓ Portfolio built successfully!
                
Portfolio Summary:
• Total Bonds: {portfolio_data.get('total_bonds', 'N/A')}
• Portfolio Value: ${portfolio_data.get('total_value', 'N/A'):,}
• Average Maturity: {portfolio_data.get('average_maturity', 'N/A')} years
• Average Yield: {portfolio_data.get('average_yield', 'N/A')}%
• Duration: {portfolio_data.get('duration', 'N/A')} years

You can now ask questions about your portfolio."""
            else:
                acknowledgment = "Portfolio building failed. Please check your criteria and try again."
                
        elif intent == "query_portfolio":
            acknowledgment = state.get("_query_response", "Portfolio information retrieved.")
            
        else:
            acknowledgment = "How can I help you build your bonds portfolio? You can specify bond universe criteria or optimization constraints."
        
        # Add AI response to messages
        state["messages"].append(AIMessage(content=acknowledgment))
        
        return state
    
    def _route_based_on_intent(self, state: PortfolioState) -> str:
        """Route to appropriate node based on intent"""
        intent = state.get("current_intent", "general")
        
        if "bond_criteria" in intent:
            return "bond_criteria"
        elif "optimization" in intent:
            return "optimization"
        elif "build_portfolio" in intent:
            return "build_portfolio"
        elif "query_portfolio" in intent:
            return "query_portfolio"
        else:
            return "general"
    
    def chat(self, message: str, state: Optional[PortfolioState] = None) -> tuple[str, PortfolioState]:
        """Main chat interface"""
        if state is None:
            state = {
                "messages": [],
                "bond_universe": [],
                "optimization_constraints": [],
                "current_intent": None,
                "portfolio_built": False,
                "portfolio_data": None
            }
        
        # Add user message to state
        state["messages"].append(HumanMessage(content=message))
        
        # Run the graph
        result = self.graph.invoke(state)
        
        # Return the latest AI message and updated state
        ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
        latest_response = ai_messages[-1].content if ai_messages else "I'm ready to help you build your portfolio."
        
        return latest_response, result


def interactive_chat():
    """Interactive command-line interface for the bond portfolio bot"""
    print("🤖 Bond Portfolio Builder")
    print("=" * 50)
    print("I'll help you build a US bonds portfolio!")
    print("You can specify:")
    print("• Bond universe criteria (maturity, sector, ticker, rating, etc.)")
    print("• Optimization constraints (duration, yield limits, etc.)")
    print("• Type 'build portfolio' when ready to create your portfolio")
    print("• Type 'status' to see current criteria")
    print("• Type 'quit' or 'exit' to end")
    print("=" * 50)
    
    # Initialize the agent
    agent = BondPortfolioAgent(
        api_key="your-openai-api-key",  # Replace with actual API key
        portfolio_api_endpoint="https://your-portfolio-api.com/build"
    )
    
    state = None
    
    while True:
        try:
            # Get user input
            user_input = input("\n💬 You: ").strip()
            
            # Handle special commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Thanks for using Bond Portfolio Builder!")
                break
            
            if user_input.lower() == 'status':
                if state:
                    print(f"\n📊 Current Status:")
                    print(f"Bond Universe Criteria: {len(state.get('bond_universe', []))}")
                    for i, criteria in enumerate(state.get('bond_universe', []), 1):
                        if hasattr(criteria, 'category'):
                            print(f"  {i}. {criteria.category}: {criteria.value}")
                        else:
                            print(f"  {i}. {criteria}")
                    
                    print(f"Optimization Constraints: {len(state.get('optimization_constraints', []))}")
                    for i, constraint in enumerate(state.get('optimization_constraints', []), 1):
                        print(f"  {i}. {constraint}")
                    
                    print(f"Portfolio Built: {'Yes' if state.get('portfolio_built', False) else 'No'}")
                else:
                    print("\n📊 No criteria captured yet. Start by specifying bond characteristics!")
                continue
            
            if not user_input:
                print("Please enter a message or type 'quit' to exit.")
                continue
            
            # Process the message through the agent
            response, state = agent.chat(user_input, state)
            
            # Display bot response
            print(f"\n🤖 Bot: {response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")
            print("Please try again or type 'quit' to exit.")


def demo_conversation():
    """Run a demo conversation with predefined messages"""
    print("🎬 Running Demo Conversation")
    print("=" * 50)
    
    # Initialize the agent
    agent = BondPortfolioAgent(
        api_key="demo-key",
        portfolio_api_endpoint="https://demo-api.com/build"
    )
    
    state = None
    
    # Example interactions
    demo_messages = [
        "I want bonds with maturity between 5-10 years",
        "Add corporate bonds from technology sector", 
        "Include treasury bonds with AAA rating",
        "Replace maturity to 3-7 years instead",
        "Set maximum duration of 6 years",
        "Build portfolio",
        "What's the average yield of my portfolio?"
    ]
    
    for message in demo_messages:
        print(f"\n💬 User: {message}")
        try:
            response, state = agent.chat(message, state)
            print(f"🤖 Bot: {response}")
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        # Pause for readability
        input("Press Enter to continue...")


def main():
    """Main function with options for interactive or demo mode"""
    print("Bond Portfolio Builder")
    print("1. Interactive Chat")
    print("2. Demo Conversation")
    print("3. Exit")
    
    while True:
        choice = input("\nSelect option (1-3): ").strip()
        
        if choice == '1':
            interactive_chat()
            break
        elif choice == '2':
            demo_conversation()
            break
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            print("Please enter 1, 2, or 3")


if __name__ == "__main__":
    main()
