import asyncio
import json
import re
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from dataclasses import dataclass, asdict
from enum import Enum
import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.bar import Bar
from rich.columns import Columns

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

console = Console()

class Direction(Enum):
    CLIENT_BUY = "Client Buy"
    CLIENT_SELL = "Client Sell"
    TWO_WAY = "2-way"

@dataclass
class PortfolioSize:
    notional_range: Optional[tuple] = None  # (min, max) in K
    dirty_market_value_range: Optional[tuple] = None  # (min, max) in K

@dataclass
class OverallCharacteristics:
    portfolio_name: Optional[str] = None
    direction: Optional[Direction] = None
    portfolio_size: PortfolioSize = None
    
    def __post_init__(self):
        if self.portfolio_size is None:
            self.portfolio_size = PortfolioSize()

@dataclass
class BondCharacteristic:
    field: str
    value: Any
    operator: str = "="  # =, >, <, >=, <=, in, not_in
    
class BondUniverse:
    def __init__(self):
        self.characteristics: List[BondCharacteristic] = []
        self.excluded_fields: List[str] = []
        self.last_api_response: Optional[Dict] = None
        
    def add_characteristic(self, characteristic: BondCharacteristic):
        # Replace if same field exists
        self.characteristics = [c for c in self.characteristics if c.field != characteristic.field]
        self.characteristics.append(characteristic)
        
    def remove_characteristic(self, field: str):
        self.characteristics = [c for c in self.characteristics if c.field != field]
        
    def clear_all(self):
        self.characteristics = []
        self.excluded_fields = []

class ChatState(TypedDict):
    messages: Annotated[List, add_messages]
    overall_characteristics: OverallCharacteristics
    bond_universe: BondUniverse
    optimization_constraints: Dict[str, Any]
    current_step: str
    api_base_url: str

class BondPortfolioAgent:
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        self.llm = ChatOpenAI(model="gpt-4", temperature=0)
        self.api_base_url = api_base_url
        
        # Create the graph
        workflow = StateGraph(ChatState)
        
        # Add nodes
        workflow.add_node("classifier", self.classify_input)
        workflow.add_node("overall_handler", self.handle_overall_characteristics)
        workflow.add_node("universe_handler", self.handle_bond_universe)
        workflow.add_node("optimization_handler", self.handle_optimization_constraints)
        workflow.add_node("api_caller", self.call_bond_api)
        workflow.add_node("visualizer", self.create_visualization)
        workflow.add_node("responder", self.generate_response)
        
        # Add edges
        workflow.set_entry_point("classifier")
        workflow.add_conditional_edges(
            "classifier",
            self.route_classification,
            {
                "overall": "overall_handler",
                "universe": "universe_handler", 
                "optimization": "optimization_handler",
                "show": "visualizer",
                "general": "responder"
            }
        )
        
        workflow.add_edge("overall_handler", "responder")
        workflow.add_edge("universe_handler", "api_caller")
        workflow.add_edge("optimization_handler", "responder")
        workflow.add_edge("api_caller", "visualizer")
        workflow.add_edge("visualizer", "responder")
        workflow.add_edge("responder", END)
        
        self.graph = workflow.compile()
        
    def classify_input(self, state: ChatState) -> ChatState:
        """Classify user input to determine which handler to use"""
        last_message = state["messages"][-1].content
        
        # Classification prompts
        classification_prompt = f"""
        Classify the following user input into one of these categories:
        - "overall": Portfolio name, direction (Client Buy/Sell/2-way), portfolio size (notional/dirty market value)
        - "universe": Bond characteristics like maturity, sectors, tickers, duration, yield, price, spread, liquidity score, etc.
        - "optimization": Optimization constraints (you'll handle this later)
        - "show": User wants to see current state, captured characteristics, or visualizations
        - "general": General questions or unclear requests
        
        User input: "{last_message}"
        
        Respond with only the category name.
        """
        
        response = self.llm.invoke([SystemMessage(content=classification_prompt)])
        classification = response.content.strip().lower()
        
        state["current_step"] = classification
        return state
    
    def route_classification(self, state: ChatState) -> str:
        """Route based on classification"""
        return state["current_step"]
    
    def handle_overall_characteristics(self, state: ChatState) -> ChatState:
        """Handle overall portfolio characteristics"""
        last_message = state["messages"][-1].content
        overall_chars = state.get("overall_characteristics", OverallCharacteristics())
        
        # Extract portfolio name
        name_match = re.search(r'(?:name|portfolio)\s*(?:is|:)?\s*([^,\n]+)', last_message, re.IGNORECASE)
        if name_match:
            overall_chars.portfolio_name = name_match.group(1).strip().strip('"\'')
        
        # Extract direction
        if any(word in last_message.lower() for word in ['client buy', 'buy']):
            overall_chars.direction = Direction.CLIENT_BUY
        elif any(word in last_message.lower() for word in ['client sell', 'sell']):
            overall_chars.direction = Direction.CLIENT_SELL
        elif '2-way' in last_message.lower() or 'two-way' in last_message.lower():
            overall_chars.direction = Direction.TWO_WAY
        
        # Extract portfolio size
        notional_match = re.search(r'notional[:\s]*(\d+(?:\.\d+)?)[kK]?\s*-\s*(\d+(?:\.\d+)?)[kK]?', last_message, re.IGNORECASE)
        if notional_match:
            min_val, max_val = float(notional_match.group(1)), float(notional_match.group(2))
            overall_chars.portfolio_size.notional_range = (min_val, max_val)
        
        dirty_mv_match = re.search(r'dirty\s+market\s+value[:\s]*(\d+(?:\.\d+)?)[kK]?\s*-\s*(\d+(?:\.\d+)?)[kK]?', last_message, re.IGNORECASE)
        if dirty_mv_match:
            min_val, max_val = float(dirty_mv_match.group(1)), float(dirty_mv_match.group(2))
            overall_chars.portfolio_size.dirty_market_value_range = (min_val, max_val)
        
        state["overall_characteristics"] = overall_chars
        
        # Create acknowledgment
        ack_parts = []
        if overall_chars.portfolio_name:
            ack_parts.append(f"Portfolio name: {overall_chars.portfolio_name}")
        if overall_chars.direction:
            ack_parts.append(f"Direction: {overall_chars.direction.value}")
        if overall_chars.portfolio_size.notional_range:
            ack_parts.append(f"Notional range: {overall_chars.portfolio_size.notional_range[0]}K - {overall_chars.portfolio_size.notional_range[1]}K")
        if overall_chars.portfolio_size.dirty_market_value_range:
            ack_parts.append(f"Dirty MV range: {overall_chars.portfolio_size.dirty_market_value_range[0]}K - {overall_chars.portfolio_size.dirty_market_value_range[1]}K")
        
        acknowledgment = "✓ Captured: " + " | ".join(ack_parts) if ack_parts else "No new characteristics captured."
        state["messages"].append(AIMessage(content=acknowledgment))
        
        return state
    
    def handle_bond_universe(self, state: ChatState) -> ChatState:
        """Handle bond universe characteristics"""
        last_message = state["messages"][-1].content
        bond_universe = state.get("bond_universe", BondUniverse())
        
        # Check if user wants to replace all characteristics
        if any(phrase in last_message.lower() for phrase in ['replace all', 'clear all', 'start over']):
            bond_universe.clear_all()
            state["messages"].append(AIMessage(content="✓ All bond universe characteristics cleared."))
            state["bond_universe"] = bond_universe
            return state
        
        # Parse characteristics from input
        characteristics_found = []
        
        # Common bond fields and their patterns
        field_patterns = {
            'maturity': r'maturity[:\s]*([^,\n]+)',
            'sector': r'sector[s]?[:\s]*([^,\n]+)',
            'ticker': r'ticker[s]?[:\s]*([^,\n]+)',
            'isin': r'isin[s]?[:\s]*([^,\n]+)',
            'cusip': r'cusip[s]?[:\s]*([^,\n]+)',
            'duration': r'duration[:\s]*([0-9.-]+)',
            'yield': r'yield[:\s]*([0-9.%-]+)',
            'price': r'price[:\s]*([0-9.-]+)',
            'spread': r'spread[:\s]*([0-9.-]+)',
            'liquidity_score': r'liquidity[:\s]*([0-9-]+)',
            'rating': r'rating[s]?[:\s]*([^,\n]+)',
            'currency': r'currency[:\s]*([A-Z]{3})',
            'country': r'country[:\s]*([^,\n]+)',
        }
        
        for field, pattern in field_patterns.items():
            match = re.search(pattern, last_message, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                # Determine operator based on context
                operator = "="
                if ">" in value:
                    operator = ">"
                    value = value.replace(">", "").strip()
                elif "<" in value:
                    operator = "<"
                    value = value.replace("<", "").strip()
                elif "-" in value and field in ['duration', 'yield', 'price', 'spread']:
                    # Range handling
                    parts = value.split("-")
                    if len(parts) == 2:
                        operator = "range"
                        value = (float(parts[0].strip()), float(parts[1].strip()))
                
                characteristic = BondCharacteristic(field=field, value=value, operator=operator)
                bond_universe.add_characteristic(characteristic)
                characteristics_found.append(f"{field}: {value}")
        
        state["bond_universe"] = bond_universe
        
        # Create acknowledgment
        if characteristics_found:
            acknowledgment = "✓ Bond universe updated: " + " | ".join(characteristics_found)
        else:
            acknowledgment = "No bond characteristics detected. Please specify characteristics like maturity, sector, rating, etc."
        
        state["messages"].append(AIMessage(content=acknowledgment))
        return state
    
    async def call_bond_api(self, state: ChatState) -> ChatState:
        """Make API call to get bonds matching criteria"""
        bond_universe = state.get("bond_universe", BondUniverse())
        
        # Prepare API payload
        criteria = {}
        for char in bond_universe.characteristics:
            if char.operator == "range":
                criteria[char.field] = {"min": char.value[0], "max": char.value[1]}
            else:
                criteria[char.field] = {"value": char.value, "operator": char.operator}
        
        # Mock API call (replace with actual API endpoint)
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base_url}/bonds/search",
                    json={"criteria": criteria},
                    timeout=30.0
                )
                if response.status_code == 200:
                    bond_universe.last_api_response = response.json()
                else:
                    # Mock response for demo
                    bond_universe.last_api_response = self._create_mock_response(criteria)
        except Exception:
            # Mock response for demo
            bond_universe.last_api_response = self._create_mock_response(criteria)
        
        state["bond_universe"] = bond_universe
        return state
    
    def _create_mock_response(self, criteria: Dict) -> Dict:
        """Create mock API response for demo purposes"""
        import random
        
        # Generate mock aggregations
        aggregations = {}
        for field in criteria.keys():
            if field == "sector":
                aggregations[field] = {
                    "Corporate": random.randint(100, 500),
                    "Government": random.randint(50, 300),
                    "Municipal": random.randint(30, 200),
                    "Treasury": random.randint(80, 400)
                }
            elif field == "rating":
                aggregations[field] = {
                    "AAA": random.randint(10, 100),
                    "AA": random.randint(20, 150),
                    "A": random.randint(50, 200),
                    "BBB": random.randint(30, 180),
                    "BB": random.randint(20, 120)
                }
            elif field == "maturity":
                aggregations[field] = {
                    "0-2 years": random.randint(50, 200),
                    "2-5 years": random.randint(100, 300),
                    "5-10 years": random.randint(80, 250),
                    "10+ years": random.randint(30, 150)
                }
        
        return {
            "total_bonds": random.randint(500, 2000),
            "aggregations": aggregations
        }
    
    def create_visualization(self, state: ChatState) -> ChatState:
        """Create rich visualizations for bond data"""
        bond_universe = state.get("bond_universe", BondUniverse())
        
        # Display current characteristics
        if bond_universe.characteristics:
            table = Table(title="Current Bond Universe Characteristics")
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="green")
            table.add_column("Operator", style="yellow")
            
            for char in bond_universe.characteristics:
                table.add_row(char.field, str(char.value), char.operator)
            
            console.print(table)
        
        # Display API response aggregations
        if bond_universe.last_api_response:
            api_data = bond_universe.last_api_response
            console.print(f"\n[bold green]Total Bonds Found: {api_data.get('total_bonds', 0)}[/bold green]")
            
            # Create bar charts for aggregations
            aggregations = api_data.get('aggregations', {})
            for field, data in aggregations.items():
                if isinstance(data, dict):
                    console.print(f"\n[bold blue]{field.title()} Distribution:[/bold blue]")
                    
                    # Create simple bar visualization
                    max_count = max(data.values()) if data else 1
                    for category, count in data.items():
                        bar_length = int((count / max_count) * 20)
                        bar = "█" * bar_length
                        console.print(f"  {category:<15} {bar} {count}")
        
        return state
    
    def handle_optimization_constraints(self, state: ChatState) -> ChatState:
        """Handle optimization constraints (placeholder for future implementation)"""
        state["messages"].append(AIMessage(content="✓ Optimization constraints handler - to be implemented with your specific requirements."))
        return state
    
    def generate_response(self, state: ChatState) -> ChatState:
        """Generate final response to user"""
        # The acknowledgments are already added in individual handlers
        # This is mainly for general queries or additional context
        
        if state["current_step"] == "general":
            last_message = state["messages"][-1].content
            
            system_prompt = """You are a precise bond portfolio assistant. 
            Provide brief, direct responses. Always acknowledge what has been captured or processed.
            Available commands:
            - Set portfolio characteristics (name, direction, size)
            - Add bond universe criteria (maturity, sector, rating, etc.)
            - Show current state
            - Optimization constraints (coming soon)
            """
            
            response = self.llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=last_message)
            ])
            
            state["messages"].append(AIMessage(content=response.content))
        
        return state

# CLI Interface using built-in input()
class BondPortfolioCLI:
    def __init__(self):
        self.agent = BondPortfolioAgent()
        self.state = {
            "messages": [],
            "overall_characteristics": OverallCharacteristics(),
            "bond_universe": BondUniverse(),
            "optimization_constraints": {},
            "current_step": "",
            "api_base_url": "http://localhost:8000"
        }
    
    def print_welcome(self):
        """Print welcome message and instructions"""
        print("\n" + "="*60)
        print("🏦 BOND PORTFOLIO ASSISTANT")
        print("="*60)
        print("\nAvailable commands:")
        print("📋 Portfolio Setup:")
        print("   - Set portfolio name, direction (Client Buy/Sell/2-way), size")
        print("   - Example: 'Portfolio name Corporate Bonds Q4, direction Client Buy, notional 1000K-5000K'")
        print("\n🔍 Bond Universe:")
        print("   - Add characteristics: maturity, sector, rating, duration, yield, etc.")
        print("   - Example: 'Add sector Corporate, rating AAA, maturity 2-10 years'")
        print("   - Replace all: 'Replace all with sector Government, duration < 5'")
        print("\n📊 View Data:")
        print("   - Type 'show' or 'display' to see current state")
        print("   - Type 'summary' for portfolio overview")
        print("\n❌ Exit: Type 'exit' or 'quit'")
        print("="*60)
    
    def print_current_state(self):
        """Print current portfolio state"""
        print("\n" + "="*50)
        print("📊 CURRENT PORTFOLIO STATE")
        print("="*50)
        
        # Overall characteristics
        overall = self.state["overall_characteristics"]
        print(f"\n📋 Portfolio Overview:")
        print(f"   Name: {overall.portfolio_name or 'Not set'}")
        print(f"   Direction: {overall.direction.value if overall.direction else 'Not set'}")
        
        if overall.portfolio_size.notional_range:
            print(f"   Notional: {overall.portfolio_size.notional_range[0]}K - {overall.portfolio_size.notional_range[1]}K")
        else:
            print(f"   Notional: Not set")
            
        if overall.portfolio_size.dirty_market_value_range:
            print(f"   Dirty MV: {overall.portfolio_size.dirty_market_value_range[0]}K - {overall.portfolio_size.dirty_market_value_range[1]}K")
        else:
            print(f"   Dirty MV: Not set")
        
        # Bond universe
        bond_universe = self.state["bond_universe"]
        print(f"\n🔍 Bond Universe ({len(bond_universe.characteristics)} characteristics):")
        if bond_universe.characteristics:
            for char in bond_universe.characteristics:
                operator_symbol = {"=": "=", ">": ">", "<": "<", "range": "between"}
                op_display = operator_symbol.get(char.operator, char.operator)
                print(f"   • {char.field}: {op_display} {char.value}")
        else:
            print("   No characteristics set")
        
        # API results summary
        if bond_universe.last_api_response:
            total = bond_universe.last_api_response.get('total_bonds', 0)
            print(f"\n📈 Search Results: {total} bonds found")
        
        print("="*50)
    
    def run(self):
        """Run the interactive CLI with synchronous processing"""
        self.print_welcome()
        
        while True:
            try:
                # Get user input
                user_input = input("\n🤖 Enter command: ").strip()
                
                # Handle exit
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Thank you for using Bond Portfolio Assistant!")
                    break
                
                # Handle empty input
                if not user_input:
                    print("💡 Please enter a command. Type 'show' to see current state.")
                    continue
                
                # Handle show commands
                if user_input.lower() in ['show', 'display', 'state', 'summary']:
                    self.print_current_state()
                    continue
                
                # Process command through the agent
                print(f"\n⚡ Processing: {user_input}")
                
                # Add user message to state
                self.state["messages"].append(HumanMessage(content=user_input))
                
                # Process through the graph (run async function synchronously)
                result = asyncio.run(self.agent.graph.ainvoke(self.state))
                self.state = result
                
                # Print the response
                if result["messages"] and isinstance(result["messages"][-1], AIMessage):
                    response = result["messages"][-1].content
                    print(f"\n✅ {response}")
                
                # Show updated state if bond universe was modified
                if self.state["current_step"] == "universe" and self.state["bond_universe"].last_api_response:
                    self.show_bond_distributions()
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("💡 Please try again with a different command.")
    
    def show_bond_distributions(self):
        """Show bond distribution charts in simple text format"""
        bond_universe = self.state["bond_universe"]
        if not bond_universe.last_api_response:
            return
        
        api_data = bond_universe.last_api_response
        aggregations = api_data.get('aggregations', {})
        
        if aggregations:
            print(f"\n📊 BOND DISTRIBUTIONS")
            print("-" * 40)
            
            for field, data in aggregations.items():
                if isinstance(data, dict):
                    print(f"\n📈 {field.title()}:")
                    
                    # Create simple text bar chart
                    max_count = max(data.values()) if data else 1
                    for category, count in data.items():
                        bar_length = int((count / max_count) * 20)
                        bar = "█" * bar_length
                        percentage = (count / sum(data.values())) * 100
                        print(f"   {category:<15} {bar:<20} {count:>4} ({percentage:.1f}%)")

# Simple test function
def test_basic_functionality():
    """Test basic agent functionality without CLI"""
    print("🧪 Testing Bond Portfolio Agent...")
    
    agent = BondPortfolioAgent()
    test_state = {
        "messages": [HumanMessage(content="Set portfolio name to 'Test Portfolio', direction Client Buy")],
        "overall_characteristics": OverallCharacteristics(),
        "bond_universe": BondUniverse(),
        "optimization_constraints": {},
        "current_step": "",
        "api_base_url": "http://localhost:8000"
    }
    
    try:
        result = asyncio.run(agent.graph.ainvoke(test_state))
        print("✅ Agent test successful!")
        if result["messages"]:
            print(f"📝 Response: {result['messages'][-1].content}")
        return True
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        return False

# Main execution
if __name__ == "__main__":
    print("🚀 Starting Bond Portfolio Assistant...")
    
    # Test the agent first
    if test_basic_functionality():
        print("\n✅ All systems ready!")
        
        # Start the CLI
        try:
            cli = BondPortfolioCLI()
            cli.run()
        except Exception as e:
            print(f"❌ CLI Error: {e}")
    else:
        print("❌ System check failed. Please check your dependencies.")
        
    print("\n🏁 Program ended.")
