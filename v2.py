#!/usr/bin/env python3
"""
Bond Portfolio Chatbot with LangGraph
Usage: python bond_portfolio_chatbot.py
"""

import asyncio
import json
import re
import os
import readline
import atexit
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
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

# Load environment variables from .env file
load_dotenv()

console = Console()

# Utility function to parse dollar amounts
def parse_dollar_amount(text: str) -> Optional[float]:
    """
    Parse dollar amounts and convert to thousands (K)
    Handles: $1M, $500K, $2.5B, $1000, 500 million, 2 billion, etc.
    Returns value in thousands (K)
    """
    if not text:
        return None
    
    # Remove commas and normalize
    text = text.replace(',', '').strip().lower()
    
    # Extract number and unit patterns
    patterns = [
        # $1M, $500K, $2.5B format
        r'\$?(\d+(?:\.\d+)?)\s*([kmb])\b',
        # $1 million, 500 thousand, 2 billion format  
        r'\$?(\d+(?:\.\d+)?)\s*(thousand|million|billion)\b',
        # Plain numbers with K/M/B suffix
        r'(\d+(?:\.\d+)?)\s*([kmb])\b',
        # Words: 500 thousand, 2 million, 1 billion
        r'(\d+(?:\.\d+)?)\s*(thousand|million|billion)\b',
        # Just plain dollar amounts: $1000
        r'\$(\d+(?:\.\d+)?)\b',
        # Just plain numbers: 1000
        r'\b(\d+(?:\.\d+)?)\b'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            try:
                number = float(match.group(1))
                
                # Determine the multiplier
                if len(match.groups()) > 1:
                    unit = match.group(2).lower()
                    multipliers = {
                        'k': 1,                    # K = thousands (our base unit)
                        'm': 1000,                 # M = millions = 1000K
                        'b': 1000000,              # B = billions = 1,000,000K
                        'thousand': 1,             # thousand = 1K
                        'million': 1000,           # million = 1000K  
                        'billion': 1000000         # billion = 1,000,000K
                    }
                    multiplier = multipliers.get(unit, 1)
                else:
                    # Plain number - assume it's already in appropriate units
                    # If > 1000, likely in dollars, convert to K
                    multiplier = 0.001 if number > 1000 else 1
                
                result = number * multiplier
                return result
                
            except (ValueError, IndexError):
                continue
    
    return None

def format_amount_in_k(amount: float) -> str:
    """Format amount in K notation - always show as K units"""
    return f"{amount}K"

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
        # For sectors, allow multiple values - don't replace, just add if not already present
        if characteristic.field == "sector":
            # Check if this exact sector already exists
            existing_sectors = [c.value for c in self.characteristics if c.field == "sector"]
            if characteristic.value not in existing_sectors:
                self.characteristics.append(characteristic)
        else:
            # For other fields, replace if same field exists
            self.characteristics = [c for c in self.characteristics if c.field != characteristic.field]
            self.characteristics.append(characteristic)
        
    def remove_characteristic(self, field: str, value: str = None):
        if value is None:
            # Remove all characteristics for this field
            self.characteristics = [c for c in self.characteristics if c.field != field]
        else:
            # Remove specific characteristic with this field and value
            self.characteristics = [c for c in self.characteristics 
                                 if not (c.field == field and c.value == value)]
    
    def get_characteristics_by_field(self, field: str) -> List[BondCharacteristic]:
        """Get all characteristics for a specific field"""
        return [c for c in self.characteristics if c.field == field]
        
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
    def __init__(self, api_base_url: str = None):
        # Get API key from environment
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")
        
        # Initialize LLM with environment variable
        self.llm = ChatOpenAI(
            model="gpt-4", 
            temperature=0,
            openai_api_key=openai_api_key
        )
        
        # Get API base URL from environment or use default
        self.api_base_url = api_base_url or os.getenv("BOND_API_BASE_URL", "http://localhost:8000")
        
        # Create the graph
        workflow = StateGraph(ChatState)
        
        # Add nodes
        workflow.add_node("classifier", self.classify_input)
        workflow.add_node("overall_handler", self.handle_overall_characteristics)
        workflow.add_node("universe_handler", self.handle_bond_universe)
        workflow.add_node("mixed_handler", self.handle_mixed_input)
        workflow.add_node("clear_handler", self.handle_clear_command)
        workflow.add_node("workflow_handler", self.handle_workflow_command)
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
                "mixed": "mixed_handler",
                "clear": "clear_handler",
                "workflow": "workflow_handler",
                "optimization": "optimization_handler",
                "show": "visualizer",
                "general": "responder"
            }
        )
        
        workflow.add_edge("overall_handler", "responder")
        workflow.add_edge("universe_handler", "api_caller")
        workflow.add_edge("mixed_handler", "api_caller")
        workflow.add_edge("clear_handler", "responder")
        workflow.add_edge("workflow_handler", "responder")
        workflow.add_edge("optimization_handler", "responder")
        workflow.add_edge("api_caller", "visualizer")
        workflow.add_edge("visualizer", "responder")
        workflow.add_edge("responder", END)
        
        self.graph = workflow.compile()
        
    def classify_input(self, state: ChatState) -> ChatState:
        """Classify user input to determine which handler to use"""
        last_message = state["messages"][-1].content
        
        # Check for different types of content in the message
        contains_rating = False
        contains_bond_keywords = False
        contains_portfolio_keywords = False
        
        # Check for rating patterns
        rating_patterns = [
            r'\b(d|c|cc|ccc[-+]?|b[-+]?|bb[-+]?|bbb[-+]?|a[-+]?|aa[-+]?|aaa)\b',
            r'\b(d|c|cc|ccc[-+]?|b[-+]?|bb[-+]?|bbb[-+]?|a[-+]?|aa[-+]?|aaa)[-–—](d|c|cc|ccc[-+]?|b[-+]?|bb[-+]?|bbb[-+]?|a[-+]?|aa[-+]?|aaa)\b'
        ]
        
        for pattern in rating_patterns:
            if re.search(pattern, last_message, re.IGNORECASE):
                contains_rating = True
                break
        
        # Check for bond universe keywords
        bond_universe_keywords = ['energy', 'sector', 'rating', 'yield', 'duration', 'maturity', 'fins', 'telecom', 'tech', 'utils', 'healthcare', 'financials']
        contains_bond_keywords = any(word in last_message.lower() for word in bond_universe_keywords)
        
        # Check for portfolio setup keywords
        portfolio_keywords = ['portfolio', 'name', 'direction', 'client buy', 'client sell', '2-way', 'notional', 'dirty market', 'size']
        contains_portfolio_keywords = any(word in last_message.lower() for word in portfolio_keywords)
        
        # Check for clear/reset commands
        clear_keywords = ['clear', 'reset', 'start over', 'clear all']
        contains_clear_keywords = any(phrase in last_message.lower() for phrase in clear_keywords)
        
        # Check for workflow visualization commands
        workflow_keywords = ['agentic workflow', 'workflow', 'show workflow', 'graph', 'langgraph']
        contains_workflow_keywords = any(phrase in last_message.lower() for phrase in workflow_keywords)
        
        # Determine classification based on content type
        if contains_clear_keywords:
            # Clear/reset command
            state["current_step"] = "clear"
            print(f"🔍 Classified as: clear (reset command)")
            return state
        elif contains_workflow_keywords:
            # Workflow visualization command
            state["current_step"] = "workflow"
            print(f"🔍 Classified as: workflow (show graph)")
            return state
        elif (contains_rating or contains_bond_keywords) and contains_portfolio_keywords:
            # Mixed input - contains both portfolio and bond characteristics
            state["current_step"] = "mixed"
            print(f"🔍 Classified as: mixed (contains both portfolio and bond criteria)")
            return state
        elif contains_portfolio_keywords:
            # Only portfolio characteristics
            state["current_step"] = "overall"
            print(f"🔍 Classified as: overall (portfolio setup)")
            return state
        elif contains_rating or contains_bond_keywords:
            # Only bond universe characteristics
            state["current_step"] = "universe"
            print(f"🔍 Classified as: universe (bond characteristics)")
            return state
        
        # Use LLM classification for other cases
        classification_prompt = f"""
        Classify the following user input into one of these categories:
        - "overall": Portfolio name, direction (Client Buy/Sell/2-way), portfolio size (notional/dirty market value)
        - "universe": Bond characteristics like maturity, sectors, tickers, duration, yield, price, spread, liquidity score, etc.
        - "optimization": Optimization constraints (you'll handle this later)
        - "show": User wants to see current state, captured characteristics, or visualizations
        - "general": General questions or unclear requests
        
        User input: "{last_message}"
        
        Respond with only the category name (no quotes, no extra text).
        """
        
        try:
            response = self.llm.invoke([SystemMessage(content=classification_prompt)])
            classification = response.content.strip().lower().replace('"', '').replace("'", "")
            
            # Ensure we have a valid classification
            valid_classifications = ["overall", "universe", "optimization", "show", "general"]
            if classification not in valid_classifications:
                classification = "general"
            
            state["current_step"] = classification
            print(f"🔍 Classified as: {classification}")
            
        except Exception as e:
            print(f"Classification error: {e}")
            state["current_step"] = "general"
            
        return state
    
    def route_classification(self, state: ChatState) -> str:
        """Route based on classification"""
        classification = state.get("current_step", "general")
        print(f"🚦 Routing to: {classification}")  # Debug info
        return classification
    
    def handle_overall_characteristics(self, state: ChatState) -> ChatState:
        """Handle overall portfolio characteristics using LLM extraction"""
        last_message = state["messages"][-1].content
        overall_chars = state.get("overall_characteristics", OverallCharacteristics())
        
        # Create extraction prompt for the LLM
        current_state = {
            "portfolio_name": overall_chars.portfolio_name,
            "direction": overall_chars.direction.value if overall_chars.direction else None,
            "notional_range": overall_chars.portfolio_size.notional_range,
            "dirty_mv_range": overall_chars.portfolio_size.dirty_market_value_range
        }
        
        extraction_prompt = f"""
        Extract ONLY portfolio setup information from this user message. DO NOT extract bond characteristics.
        
        Current portfolio state: {current_state}
        
        User message: "{last_message}"
        
        Please extract and return ONLY the following PORTFOLIO information in JSON format:
        {{
            "portfolio_name": "extracted name or null if not mentioned",
            "direction": "Client Buy" or "Client Sell" or "2-way" or null if not mentioned,
            "notional_min": number or null (in thousands K - convert from any dollar notation),
            "notional_max": number or null (in thousands K - convert from any dollar notation),
            "dirty_mv_min": number or null (in thousands K - convert from any dollar notation),
            "dirty_mv_max": number or null (in thousands K - convert from any dollar notation),
            "action": "set" or "update" or "rename" (what the user wants to do)
        }}
        
        CRITICAL RULES:
        - Extract exact portfolio name without extra words like "to be", "to", etc.
        - For direction: recognize variations like "buy", "sell", "two way", "2way", etc.
        - ONLY extract dollar amounts that are explicitly for PORTFOLIO SIZE:
          * Look for keywords: "notional", "dirty market value", "dirty mv", "portfolio size"
          * IGNORE: "yield", "duration", "maturity" - these are BOND characteristics, NOT portfolio size
          * IGNORE: Any ranges that follow bond terms like "aaa yield 1-10" (this is yield range, not portfolio size)
        - For portfolio size ranges: handle formats like "notional 1000-5000", "$1M to $5M notional", "portfolio size between 500K and 2M"
        - If user says "rename" or "change name", set action to "rename"
        - If user provides new values, set action to "update"
        - Handle typos and natural language variations
        - Return null for fields not mentioned in the message
        
        EXAMPLES of what NOT to extract as portfolio size:
        - "aaa yield 1-10" → yield is NOT portfolio size
        - "duration 2-5" → duration is NOT portfolio size  
        - "rating bb-aa" → rating is NOT portfolio size
        - "maturity 5-10 years" → maturity is NOT portfolio size
        
        EXAMPLES of what TO extract as portfolio size:
        - "notional 1M-5M" → extract as notional range
        - "dirty market value 500K-2M" → extract as dirty_mv range
        - "portfolio size between 1000K and 5000K" → extract as notional range
        
        Return only valid JSON, no other text.
        """
        
        try:
            response = self.llm.invoke([SystemMessage(content=extraction_prompt)])
            
            # Parse the JSON response
            import json
            extracted_data = json.loads(response.content.strip())
            
            # Apply extracted data to the portfolio characteristics
            changes_made = []
            
            # Handle portfolio name
            if extracted_data.get("portfolio_name"):
                old_name = overall_chars.portfolio_name
                overall_chars.portfolio_name = extracted_data["portfolio_name"]
                if extracted_data.get("action") == "rename" and old_name:
                    changes_made.append(f"Portfolio renamed from '{old_name}' to '{overall_chars.portfolio_name}'")
                else:
                    changes_made.append(f"Portfolio name: {overall_chars.portfolio_name}")
            
            # Handle direction
            if extracted_data.get("direction"):
                direction_map = {
                    "Client Buy": Direction.CLIENT_BUY,
                    "Client Sell": Direction.CLIENT_SELL,
                    "2-way": Direction.TWO_WAY
                }
                if extracted_data["direction"] in direction_map:
                    overall_chars.direction = direction_map[extracted_data["direction"]]
                    changes_made.append(f"Direction: {overall_chars.direction.value}")
            
            # Handle notional range
            notional_min = extracted_data.get("notional_min")
            notional_max = extracted_data.get("notional_max")
            if notional_min is not None and notional_max is not None:
                overall_chars.portfolio_size.notional_range = (float(notional_min), float(notional_max))
                changes_made.append(f"Notional range: {format_amount_in_k(notional_min)} - {format_amount_in_k(notional_max)}")
            elif notional_min is not None or notional_max is not None:
                # Handle single value updates
                current_range = overall_chars.portfolio_size.notional_range or (0, 0)
                new_min = notional_min if notional_min is not None else current_range[0]
                new_max = notional_max if notional_max is not None else current_range[1]
                overall_chars.portfolio_size.notional_range = (float(new_min), float(new_max))
                changes_made.append(f"Notional range updated: {format_amount_in_k(new_min)} - {format_amount_in_k(new_max)}")
            
            # Handle dirty market value range
            dirty_mv_min = extracted_data.get("dirty_mv_min")
            dirty_mv_max = extracted_data.get("dirty_mv_max")
            if dirty_mv_min is not None and dirty_mv_max is not None:
                overall_chars.portfolio_size.dirty_market_value_range = (float(dirty_mv_min), float(dirty_mv_max))
                changes_made.append(f"Dirty MV range: {format_amount_in_k(dirty_mv_min)} - {format_amount_in_k(dirty_mv_max)}")
            elif dirty_mv_min is not None or dirty_mv_max is not None:
                # Handle single value updates
                current_range = overall_chars.portfolio_size.dirty_market_value_range or (0, 0)
                new_min = dirty_mv_min if dirty_mv_min is not None else current_range[0]
                new_max = dirty_mv_max if dirty_mv_max is not None else current_range[1]
                overall_chars.portfolio_size.dirty_market_value_range = (float(new_min), float(new_max))
                changes_made.append(f"Dirty MV range updated: {format_amount_in_k(new_min)} - {format_amount_in_k(new_max)}")
            
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            # Fallback: if LLM extraction fails, provide helpful message
            changes_made = ["Could not parse portfolio information. Please try rephrasing your request."]
            print(f"LLM extraction error: {e}")  # Debug info
        
        state["overall_characteristics"] = overall_chars
        
        # Create acknowledgment
        if changes_made:
            acknowledgment = "✓ " + " | ".join(changes_made)
        else:
            acknowledgment = "No portfolio characteristics detected. Please specify name, direction, or size information."
        
        state["messages"].append(AIMessage(content=acknowledgment))
        return state
    
    def extract_sectors_from_text(self, text: str) -> List[str]:
        """Pre-process text to extract all sectors before LLM processing"""
        sectors_found = []
        
        # Define sector mappings (including abbreviations)
        sector_mappings = {
            'energy': 'energy',
            'fins': 'financials',
            'fin': 'financials', 
            'financials': 'financials',
            'tech': 'technology',
            'technology': 'technology',
            'telecom': 'telecommunications',
            'telecommunications': 'telecommunications',
            'utils': 'utilities',
            'utilities': 'utilities',
            'cons': 'consumer',
            'consumer': 'consumer',
            'healthcare': 'healthcare',
            'health': 'healthcare',
            'indus': 'industrials',
            'industrials': 'industrials',
            'mats': 'materials',
            'materials': 'materials',
            'reits': 'real estate',
            'real estate': 'real estate',
            'govt': 'government',
            'government': 'government'
        }
        
        # Define rating patterns to exclude from sector parsing
        rating_patterns = [
            r'\b(d|c|cc|ccc[-+]?|b[-+]?|bb[-+]?|bbb[-+]?|a[-+]?|aa[-+]?|aaa)\b',
            r'\b(d|c|cc|ccc[-+]?|b[-+]?|bb[-+]?|bbb[-+]?|a[-+]?|aa[-+]?|aaa)[-–—](d|c|cc|ccc[-+]?|b[-+]?|bb[-+]?|bbb[-+]?|a[-+]?|aa[-+]?|aaa)\b'
        ]
        
        # Convert to lowercase for matching
        text_lower = text.lower()
        
        # Remove rating patterns from text before sector extraction
        text_for_sectors = text_lower
        for pattern in rating_patterns:
            text_for_sectors = re.sub(pattern, '', text_for_sectors, flags=re.IGNORECASE)
        
        # Remove commas and split by spaces to get individual words
        words = re.split(r'[,\s]+', text_for_sectors)
        
        # Check each word against sector mappings
        for word in words:
            word = word.strip()
            if word in sector_mappings:
                full_sector_name = sector_mappings[word]
                if full_sector_name not in sectors_found:
                    sectors_found.append(full_sector_name)
        
        return sectors_found

    def handle_bond_universe(self, state: ChatState) -> ChatState:
        """Handle bond universe characteristics using LLM extraction"""
        last_message = state["messages"][-1].content
        bond_universe = state.get("bond_universe", BondUniverse())
        
        # Check for different types of operations
        if any(phrase in last_message.lower() for phrase in ['replace all', 'clear all', 'start over']):
            bond_universe.clear_all()
            state["messages"].append(AIMessage(content="✓ All bond universe characteristics cleared."))
            state["bond_universe"] = bond_universe
            return state
        
        # Check for remove operations
        if any(phrase in last_message.lower() for phrase in ['remove', 'delete', 'drop']):
            return self._handle_remove_characteristics(state, last_message, bond_universe)
        
        # Check for replace operations  
        if any(phrase in last_message.lower() for phrase in ['replace', 'change', 'update']):
            return self._handle_replace_characteristics(state, last_message, bond_universe)
        
        # Default: Add new characteristics
        return self._handle_add_characteristics(state, last_message, bond_universe)
    
    def _handle_remove_characteristics(self, state: ChatState, message: str, bond_universe: BondUniverse) -> ChatState:
        """Handle removal of characteristics"""
        removed_items = []
        
        # Extract what to remove using LLM
        removal_prompt = f"""
        Extract what the user wants to remove from their bond universe criteria.
        
        User message: "{message}"
        
        Return JSON with items to remove:
        {{
            "remove_items": [
                {{
                    "field": "sector|rating|yield|duration|maturity|etc",
                    "value": "specific value to remove or null to remove entire field"
                }}
            ]
        }}
        
        Examples:
        - "remove energy" -> {{"field": "sector", "value": "energy"}}
        - "remove yield" -> {{"field": "yield", "value": null}}
        - "delete rating BBB" -> {{"field": "rating", "value": "BBB"}}
        - "drop fins" -> {{"field": "sector", "value": "financials"}}
        
        Map abbreviations: fins->financials, tech->technology, utils->utilities
        
        Return only valid JSON.
        """
        
        try:
            response = self.llm.invoke([SystemMessage(content=removal_prompt)])
            response_text = response.content.strip()
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            import json
            removal_data = json.loads(response_text)
            
            for item in removal_data.get("remove_items", []):
                field = item.get("field")
                value = item.get("value")
                
                if field:
                    if value is None:
                        # Remove entire field
                        before_count = len([c for c in bond_universe.characteristics if c.field == field])
                        bond_universe.remove_characteristic(field)
                        after_count = len([c for c in bond_universe.characteristics if c.field == field])
                        if before_count > after_count:
                            removed_items.append(f"all {field} criteria")
                    else:
                        # Remove specific value
                        before_count = len(bond_universe.characteristics)
                        bond_universe.remove_characteristic(field, value)
                        after_count = len(bond_universe.characteristics)
                        if before_count > after_count:
                            removed_items.append(f"{field}: {value}")
        
        except Exception as e:
            print(f"Remove parsing error: {e}")
            removed_items = ["Could not parse removal request"]
        
        # Update state with modified bond universe
        state["bond_universe"] = bond_universe
        
        # Refresh bond data and distributions
        if removed_items and removed_items != ["Could not parse removal request"]:
            try:
                # Generate new mock response with updated characteristics
                bond_universe.last_api_response = self._create_simple_mock_response(bond_universe)
                print("✅ Distributions updated after removal")
            except Exception as e:
                print(f"❌ Failed to update distributions: {e}")
        
        # Create acknowledgment
        if removed_items:
            acknowledgment = "✓ Removed: " + " | ".join(removed_items)
        else:
            acknowledgment = "No characteristics were removed. Please specify what to remove."
        
        state["messages"].append(AIMessage(content=acknowledgment))
        return state
    
    def _handle_replace_characteristics(self, state: ChatState, message: str, bond_universe: BondUniverse) -> ChatState:
        """Handle replacement of characteristics"""
        
        # Extract replacement instructions using LLM
        replacement_prompt = f"""
        Extract what the user wants to replace in their bond universe criteria.
        
        User message: "{message}"
        
        Return JSON with replacement instructions:
        {{
            "replacements": [
                {{
                    "field": "sector|rating|yield|duration|maturity|etc",
                    "old_value": "value to replace or null for entire field",
                    "new_value": "new value",
                    "operator": "=|range|>|<"
                }}
            ]
        }}
        
        Examples:
        - "replace energy with healthcare" -> {{"field": "sector", "old_value": "energy", "new_value": "healthcare", "operator": "="}}
        - "replace yield with 2-8" -> {{"field": "yield", "old_value": null, "new_value": [2, 8], "operator": "range"}}
        - "change rating to AAA" -> {{"field": "rating", "old_value": null, "new_value": "AAA", "operator": "="}}
        
        Map abbreviations: fins->financials, tech->technology, utils->utilities
        For ranges like "2-8", use operator "range" and value as [min, max]
        
        Return only valid JSON.
        """
        
        try:
            response = self.llm.invoke([SystemMessage(content=replacement_prompt)])
            response_text = response.content.strip()
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            
            import json
            replacement_data = json.loads(response_text)
            
            replaced_items = []
            
            for item in replacement_data.get("replacements", []):
                field = item.get("field")
                old_value = item.get("old_value")
                new_value = item.get("new_value")
                operator = item.get("operator", "=")
                
                if field and new_value is not None:
                    # Remove old value(s)
                    if old_value is None:
                        # Replace entire field
                        bond_universe.remove_characteristic(field)
                        replaced_items.append(f"replaced all {field} criteria")
                    else:
                        # Replace specific value
                        bond_universe.remove_characteristic(field, old_value)
                        replaced_items.append(f"replaced {field}: {old_value}")
                    
                    # Add new characteristic
                    new_char = BondCharacteristic(field=field, value=new_value, operator=operator)
                    bond_universe.add_characteristic(new_char)
                    
                    # Format display value
                    if operator == "range" and isinstance(new_value, list) and len(new_value) == 2:
                        display_value = f"{new_value[0]} - {new_value[1]}"
                    else:
                        display_value = str(new_value)
                    
                    replaced_items.append(f"with {field}: {display_value}")
        
        except Exception as e:
            print(f"Replace parsing error: {e}")
            replaced_items = ["Could not parse replacement request"]
        
        # Update state with modified bond universe
        state["bond_universe"] = bond_universe
        
        # Refresh bond data and distributions
        if replaced_items and replaced_items != ["Could not parse replacement request"]:
            try:
                # Generate new mock response with updated characteristics
                bond_universe.last_api_response = self._create_simple_mock_response(bond_universe)
                print("✅ Distributions updated after replacement")
            except Exception as e:
                print(f"❌ Failed to update distributions: {e}")
        
        # Create acknowledgment
        if replaced_items:
            acknowledgment = "✓ " + " ".join(replaced_items)
        else:
            acknowledgment = "No characteristics were replaced. Please specify what to replace."
        
        state["messages"].append(AIMessage(content=acknowledgment))
        return state
    
    def _handle_add_characteristics(self, state: ChatState, message: str, bond_universe: BondUniverse) -> ChatState:
        """Handle adding new characteristics (existing logic)"""
        # First, extract sectors using our custom parser
        extracted_sectors = self.extract_sectors_from_text(message)
        
        # Create extraction prompt for the LLM (for non-sector characteristics)
        current_characteristics = [
            {"field": char.field, "value": char.value, "operator": char.operator}
            for char in bond_universe.characteristics
        ]
        
        extraction_prompt = f"""
        Extract bond universe characteristics from this user message. Parse natural language bond criteria.
        
        Current bond characteristics: {current_characteristics}
        
        User message: "{message}"
        
        NOTE: Sectors have already been extracted separately: {extracted_sectors}
        DO NOT extract sectors - focus on other characteristics only.
        
        Please extract and return ONLY the following information in JSON format:
        {{
            "characteristics": [
                {{
                    "field": "rating|maturity|duration|yield|price|spread|liquidity_score|ticker|isin|cusip|currency|country|amount_outstanding|original_issue_size",
                    "value": "extracted value (can be single value, range, or list)",
                    "operator": "=|>|<|>=|<=|range|in|between"
                }}
            ]
        }}
        
        CRITICAL RULES for RATINGS:
        Automatically detect rating patterns without needing "rating" keyword:
        - Valid ratings: D, C, CC, CCC-, CCC, CCC+, B-, B, B+, BB-, BB, BB+, BBB-, BBB, BBB+, A-, A, A+, AA-, AA, AA+, AAA
        - Rating ranges: "aa-aaa" means AA rating to AAA rating range
        - Rating ranges: "bb--aa" means BB- rating to AA rating range  
        - Single ratings: "aa" means AA rating exactly, "bbb+" means BBB+ rating exactly
        - Examples:
          * "aa-aaa" = rating range from AA to AAA (use {{"field": "rating", "value": ["AA", "AAA"], "operator": "range"}})
          * "bb+-a-" = rating range from BB+ to A- (use {{"field": "rating", "value": ["BB+", "A-"], "operator": "range"}})
          * "aaa" = exactly AAA rating (use {{"field": "rating", "value": "AAA", "operator": "="}})
          * "bbb-" = exactly BBB- rating (use {{"field": "rating", "value": "BBB-", "operator": "="}})
        
        Other extraction rules:
        - YIELD: Extract yield ranges like "1-10", "5%", "above 3%" 
        - MATURITY: Extract maturity like "5-10 years", "short term"
        - DURATION: Extract duration like "1-4", "2-5"
        - LIQUIDITY_SCORE: Extract scores like "high liquidity", "score > 7"
        
        For ranges, use "range" operator and provide [min, max] as value.
        For single values, use "=" operator.
        
        Return only valid JSON, no other text.
        """
        
        try:
            response = self.llm.invoke([SystemMessage(content=extraction_prompt)])
            
            # Clean and parse the JSON response
            response_text = response.content.strip()
            
            # Remove any markdown formatting if present
            if response_text.startswith("```json"):
                response_text = response_text.replace("```json", "").replace("```", "").strip()
            elif response_text.startswith("```"):
                response_text = response_text.replace("```", "").strip()
            
            # Try to parse JSON
            try:
                extracted_data = json.loads(response_text)
            except json.JSONDecodeError as json_error:
                print(f"JSON parsing error: {json_error}")
                print(f"Raw LLM response: {response_text}")
                
                # Try to extract JSON from the response if it's embedded in text
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    try:
                        extracted_data = json.loads(json_match.group(0))
                        print("✅ Successfully extracted JSON from embedded text")
                    except json.JSONDecodeError:
                        print("❌ Could not parse embedded JSON either")
                        extracted_data = {"characteristics": []}
                else:
                    print("❌ No JSON found in response")
                    extracted_data = {"characteristics": []}
            
            # Apply extracted characteristics
            characteristics_found = []
            
            # First add all extracted sectors
            for sector in extracted_sectors:
                characteristic = BondCharacteristic(field="sector", value=sector, operator="=")
                bond_universe.add_characteristic(characteristic)
                characteristics_found.append(f"sector: {sector}")
            
            # Then add other characteristics from LLM
            for char_data in extracted_data.get("characteristics", []):
                field = char_data.get("field")
                value = char_data.get("value")
                operator = char_data.get("operator", "=")
                
                if field and value is not None:
                    # Create the characteristic
                    characteristic = BondCharacteristic(field=field, value=value, operator=operator)
                    bond_universe.add_characteristic(characteristic)
                    
                    # Format display value
                    if operator == "range" and isinstance(value, list) and len(value) == 2:
                        display_value = f"{value[0]} - {value[1]}"
                    elif operator == "in" and isinstance(value, list):
                        display_value = ", ".join(str(v) for v in value)
                    else:
                        display_value = str(value)
                    
                    characteristics_found.append(f"{field}: {display_value}")
            
        except Exception as e:
            # Fallback: still process sectors even if LLM fails
            characteristics_found = []
            
            # Add extracted sectors even in error case
            for sector in extracted_sectors:
                characteristic = BondCharacteristic(field="sector", value=sector, operator="=")
                bond_universe.add_characteristic(characteristic)
                characteristics_found.append(f"sector: {sector}")
            
            if not characteristics_found:
                characteristics_found = [f"Error processing characteristics: {str(e)}"]
            
            print(f"LLM extraction error: {e}")  # Debug info
        
        state["bond_universe"] = bond_universe
        
        # Create acknowledgment
        if characteristics_found:
            acknowledgment = "✓ Bond universe updated: " + " | ".join(characteristics_found)
        else:
            acknowledgment = "No bond characteristics detected. Please specify characteristics like rating, sector, yield, etc."
        
        state["messages"].append(AIMessage(content=acknowledgment))
        return state

    def handle_mixed_input(self, state: ChatState) -> ChatState:
        """Handle input that contains both portfolio and bond universe characteristics"""
        last_message = state["messages"][-1].content
        
        # First, handle portfolio characteristics
        portfolio_state = self.handle_overall_characteristics(state)
        
        # Then, handle bond universe characteristics  
        universe_state = self._handle_add_characteristics(portfolio_state, last_message, portfolio_state["bond_universe"])
        
        # Combine acknowledgments
        portfolio_msg = None
        universe_msg = None
        
        # Get the acknowledgment messages
        if len(universe_state["messages"]) >= 2:
            if isinstance(universe_state["messages"][-1], AIMessage):
                universe_msg = universe_state["messages"][-1].content
                universe_state["messages"].pop()  # Remove the individual message
            
            if isinstance(universe_state["messages"][-1], AIMessage):
                portfolio_msg = universe_state["messages"][-1].content
                universe_state["messages"].pop()  # Remove the individual message
        
        # Create combined acknowledgment
        combined_parts = []
        if portfolio_msg and "✓" in portfolio_msg:
            portfolio_part = portfolio_msg.replace("✓ Captured: ", "").replace("✓ ", "")
            combined_parts.append(f"Portfolio: {portfolio_part}")
        
        if universe_msg and "✓" in universe_msg:
            universe_part = universe_msg.replace("✓ Bond universe updated: ", "").replace("✓ ", "")
            combined_parts.append(f"Bond criteria: {universe_part}")
        
        if combined_parts:
            combined_acknowledgment = "✓ " + " | ".join(combined_parts)
        else:
            combined_acknowledgment = "✓ Mixed input processed"
        
        universe_state["messages"].append(AIMessage(content=combined_acknowledgment))
        return universe_state

    def handle_clear_command(self, state: ChatState) -> ChatState:
        """Handle clear/reset commands to clear all data"""
        # Reset all characteristics to initial state
        state["overall_characteristics"] = OverallCharacteristics()
        state["bond_universe"] = BondUniverse()
        state["optimization_constraints"] = {}
        
        # Clear any API response data
        state["bond_universe"].last_api_response = None
        
        acknowledgment = "✓ All portfolio and bond criteria cleared. Starting fresh!"
        state["messages"].append(AIMessage(content=acknowledgment))
        
        return state

    def handle_workflow_command(self, state: ChatState) -> ChatState:
        """Handle workflow visualization command"""
        
        # Create a visual representation of the LangGraph workflow
        workflow_diagram = """
🏗️  BOND PORTFOLIO AGENTIC WORKFLOW (LangGraph)
════════════════════════════════════════════════════════

                            📥 USER INPUT
                                  │
                                  ▼
                            🧠 CLASSIFIER
                        (Analyze input type)
                                  │
                    ┌─────────────┼─────────────┐
                    │             │             │
                    ▼             ▼             ▼
            📋 OVERALL      🔍 UNIVERSE     🎯 MIXED
           (Portfolio)   (Bond Criteria)  (Both Types)
                │             │             │
                │             ▼             ▼
                │        🌐 API CALLER ◄────┘
                │             │
                │             ▼
                │       📊 VISUALIZER
                │             │
                └─────────────┼─────────────┘
                              ▼
                        💬 RESPONDER
                              │
                              ▼
                            🏁 END

📍 ADDITIONAL NODES:
   🧹 CLEAR HANDLER    → 💬 RESPONDER
   🏗️  WORKFLOW HANDLER → 💬 RESPONDER  
   ⚙️  OPTIMIZATION    → 💬 RESPONDER
   📋 SHOW HANDLER     → 📊 VISUALIZER

🔄 WORKFLOW CAPABILITIES:

1️⃣  SMART CLASSIFICATION:
   • Detects portfolio setup (name, direction, size)
   • Recognizes bond criteria (sectors, ratings, yield)
   • Identifies mixed input (both portfolio + bonds)
   • Handles special commands (clear, show, workflow)

2️⃣  PORTFOLIO MANAGEMENT:
   • Portfolio name and direction
   • Notional and dirty market value ranges
   • Dollar amount parsing ($1M, 500K, 2B, etc.)

3️⃣  BOND UNIVERSE BUILDING:
   • Multi-sector support (energy, fins, tech, etc.)
   • Rating ranges (AA-AAA, BB-A+, single ratings)
   • Yield, duration, maturity ranges
   • Add, remove, replace operations

4️⃣  DYNAMIC VISUALIZATIONS:
   • Real-time bond distributions
   • Colored bar charts by characteristic
   • Only shows selected criteria
   • Updates on any modification

5️⃣  INTELLIGENT PROCESSING:
   • Natural language understanding
   • Typo tolerance and abbreviations
   • Context-aware parsing
   • Automatic API integration

🎯 SUPPORTED COMMANDS:
   • "portfolio name Tech Bonds, energy fins, aa-aaa"
   • "replace yield with 3-7"
   • "remove energy"
   • "clear" / "show" / "agentic workflow"
"""
        
        console.print(Panel(workflow_diagram, title="🏗️ LangGraph Agentic Workflow", border_style="bright_blue"))
        
        acknowledgment = "✓ LangGraph workflow diagram displayed above"
        state["messages"].append(AIMessage(content=acknowledgment))
        
        return state

    def handle_optimization_constraints(self, state: ChatState) -> ChatState:
        """Handle optimization constraints (placeholder for future implementation)"""
        state["messages"].append(AIMessage(content="✓ Optimization constraints handler - to be implemented with your specific requirements."))
        return state

    async def call_bond_api(self, state: ChatState) -> ChatState:
        """Make API call to get bonds matching criteria"""
        bond_universe = state.get("bond_universe", BondUniverse())
        
        # Prepare API payload - handle multiple values for same field
        api_criteria = {}
        for char in bond_universe.characteristics:
            if char.field in api_criteria:
                # Handle multiple values for same field (like multiple sectors)
                if not isinstance(api_criteria[char.field], list):
                    api_criteria[char.field] = [api_criteria[char.field]]
                if char.operator == "range":
                    api_criteria[char.field].append({"min": char.value[0], "max": char.value[1]})
                else:
                    api_criteria[char.field].append({"value": char.value, "operator": char.operator})
            else:
                # First value for this field
                if char.operator == "range":
                    api_criteria[char.field] = {"min": char.value[0], "max": char.value[1]}
                else:
                    api_criteria[char.field] = {"value": char.value, "operator": char.operator}
        
        # Mock API call (replace with actual API endpoint)
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.api_base_url}/bonds/search",
                    json={"criteria": api_criteria},
                    timeout=30.0
                )
                if response.status_code == 200:
                    bond_universe.last_api_response = response.json()
                else:
                    # Mock response for demo
                    bond_universe.last_api_response = self._create_simple_mock_response(bond_universe)
        except Exception as e:
            # Mock response for demo
            print(f"API call failed: {e}, using mock data")
            try:
                bond_universe.last_api_response = self._create_simple_mock_response(bond_universe)
                print("✅ Simple mock response generated successfully")
            except Exception as mock_error:
                print(f"❌ Mock response generation failed: {mock_error}")
                # Create minimal fallback response
                bond_universe.last_api_response = {
                    "total_bonds": 1000,
                    "aggregations": {}
                }
        
        state["bond_universe"] = bond_universe
        return state

    def _create_simple_mock_response(self, bond_universe: BondUniverse) -> Dict:
        """Create a simple mock response for testing"""
        import random
        
        aggregations = {}
        
        # Process each field type separately to ensure clean replacement
        fields_present = set(char.field for char in bond_universe.characteristics)
        
        # Handle sectors
        if "sector" in fields_present:
            aggregations["sector"] = {}
            for char in bond_universe.characteristics:
                if char.field == "sector":
                    aggregations["sector"][char.value.title()] = random.randint(100, 500)
        
        # Handle yield - only process the latest yield characteristic
        if "yield" in fields_present:
            yield_chars = [char for char in bond_universe.characteristics if char.field == "yield"]
            if yield_chars:
                # Take the last (most recent) yield characteristic
                latest_yield = yield_chars[-1]
                
                print(f"🔄 Creating yield distribution for: {latest_yield.value} (operator: {latest_yield.operator})")
                
                if latest_yield.operator == "range" and isinstance(latest_yield.value, list) and len(latest_yield.value) == 2:
                    min_yield, max_yield = latest_yield.value[0], latest_yield.value[1]
                    yield_range = max_yield - min_yield
                    
                    if yield_range <= 3:
                        # Small range: create 2 buckets
                        aggregations["yield"] = {
                            f"{min_yield:.1f}-{min_yield + yield_range/2:.1f}%": random.randint(100, 300),
                            f"{min_yield + yield_range/2:.1f}-{max_yield:.1f}%": random.randint(150, 400)
                        }
                    else:
                        # Larger range: create 3 buckets
                        bucket_size = yield_range / 3
                        aggregations["yield"] = {
                            f"{min_yield:.1f}-{min_yield + bucket_size:.1f}%": random.randint(100, 300),
                            f"{min_yield + bucket_size:.1f}-{min_yield + 2*bucket_size:.1f}%": random.randint(150, 400),
                            f"{min_yield + 2*bucket_size:.1f}-{max_yield:.1f}%": random.randint(120, 350)
                        }
                elif latest_yield.operator == "=" and latest_yield.value:
                    # Single yield value
                    aggregations["yield"] = {f"{latest_yield.value}%": random.randint(200, 500)}
        
        # Handle rating - only process the latest rating characteristic
        if "rating" in fields_present:
            rating_chars = [char for char in bond_universe.characteristics if char.field == "rating"]
            if rating_chars:
                # Take the last (most recent) rating characteristic
                latest_rating = rating_chars[-1]
                
                print(f"🔄 Creating rating distribution for: {latest_rating.value} (operator: {latest_rating.operator})")
                
                if latest_rating.operator == "range" and isinstance(latest_rating.value, list) and len(latest_rating.value) == 2:
                    # Create distribution within the rating range
                    start_rating, end_rating = latest_rating.value[0], latest_rating.value[1]
                    ratings_in_range = self._get_ratings_in_range(start_rating, end_rating)
                    rating_dist = {}
                    for rating in ratings_in_range:
                        rating_dist[rating] = random.randint(50, 300)
                    aggregations["rating"] = rating_dist
                    print(f"📊 Rating range {start_rating}-{end_rating} includes: {ratings_in_range}")
                elif latest_rating.operator == "=" and latest_rating.value:
                    # Single rating
                    aggregations["rating"] = {latest_rating.value.upper(): random.randint(200, 500)}
                    print(f"📊 Single rating: {latest_rating.value.upper()}")
        
        # Handle duration - only process the latest duration characteristic
        if "duration" in fields_present:
            duration_chars = [char for char in bond_universe.characteristics if char.field == "duration"]
            if duration_chars:
                latest_duration = duration_chars[-1]
                if latest_duration.operator == "range" and isinstance(latest_duration.value, list) and len(latest_duration.value) == 2:
                    min_dur, max_dur = latest_duration.value[0], latest_duration.value[1]
                    aggregations["duration"] = {
                        f"{min_dur}-{min_dur+1} years": random.randint(90, 220),
                        f"{min_dur+1}-{max_dur} years": random.randint(120, 280)
                    }
                else:
                    aggregations["duration"] = {
                        "1-3 years": random.randint(90, 220),
                        "3-5 years": random.randint(120, 280)
                    }
        
        # Only show the exact sectors the user selected - no automatic sub-sectors
        # This keeps the distribution clean and focused on user's actual selection
        
        total_bonds = random.randint(500, 2000)
        print(f"📊 Generated mock response: {total_bonds} bonds, {len(aggregations)} field types")
        
        return {
            "total_bonds": total_bonds,
            "aggregations": aggregations
        }

    def _get_ratings_in_range(self, start_rating: str, end_rating: str) -> List[str]:
        """Get list of ratings between start and end rating"""
        # Complete rating scale (highest to lowest quality)
        rating_scale = [
            "AAA", "AA+", "AA", "AA-",
            "A+", "A", "A-", 
            "BBB+", "BBB", "BBB-",
            "BB+", "BB", "BB-",
            "B+", "B", "B-",
            "CCC+", "CCC", "CCC-",
            "CC", "C", "D"
        ]
        
        # Normalize ratings to uppercase
        start_rating = start_rating.upper()
        end_rating = end_rating.upper()
        
        try:
            start_idx = rating_scale.index(start_rating)
            end_idx = rating_scale.index(end_rating)
            
            # Return ratings in the range (start_idx to end_idx inclusive)
            # Handle both directions (high to low, low to high)
            if start_idx <= end_idx:
                return rating_scale[start_idx:end_idx+1]
            else:
                return rating_scale[end_idx:start_idx+1]
        except ValueError:
            # If ratings not found in scale, return a default set
            print(f"Warning: Could not find ratings {start_rating} or {end_rating} in scale")
            return ["BBB", "BBB-", "BB+", "BB", "BB-"]

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
            
            # Create colorful bar charts for aggregations
            aggregations = api_data.get('aggregations', {})
            
            # Define color schemes for different chart types
            color_schemes = {
                'sector': ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan'],
                'rating': ['bright_green', 'green', 'yellow', 'orange', 'red', 'bright_red'],
                'maturity': ['bright_blue', 'blue', 'purple', 'magenta'],
                'default': ['bright_cyan', 'bright_magenta', 'bright_yellow', 'bright_green', 'bright_red', 'bright_blue']
            }
            
            for field, data in aggregations.items():
                if isinstance(data, dict):
                    console.print(f"\n[bold blue]{field.title()} Distribution:[/bold blue]")
                    
                    # Get appropriate color scheme
                    colors = color_schemes.get(field, color_schemes['default'])
                    
                    # Create colorful bar visualization
                    max_count = max(data.values()) if data else 1
                    total_count = sum(data.values())
                    
                    for i, (category, count) in enumerate(data.items()):
                        # Calculate bar length and percentage
                        bar_length = max(1, int((count / max_count) * 30))  # Increased length for better visibility
                        percentage = (count / total_count) * 100
                        
                        # Select color from scheme (cycle through if more items than colors)
                        color = colors[i % len(colors)]
                        
                        # Create colored bar using Rich markup
                        colored_bar = f"[{color}]{'█' * bar_length}[/{color}]"
                        
                        # Format the output with Rich styling
                        console.print(f"  [white]{category:<15}[/white] {colored_bar} [bright_white]{count:>4}[/bright_white] [dim]({percentage:.1f}%)[/dim]")
        
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
        try:
            self.agent = BondPortfolioAgent()
        except ValueError as e:
            print(f"❌ Configuration Error: {e}")
            print("💡 Please create a .env file with your OPENAI_API_KEY")
            raise
            
        self.state = {
            "messages": [],
            "overall_characteristics": OverallCharacteristics(),
            "bond_universe": BondUniverse(),
            "optimization_constraints": {},
            "current_step": "",
            "api_base_url": self.agent.api_base_url
        }
        
        # Setup command history
        self.setup_command_history()
    
    def setup_command_history(self):
        """Setup readline for command history and arrow key navigation"""
        try:
            # Set up history file
            history_file = os.path.expanduser("~/.bond_portfolio_history")
            
            # Load existing history
            try:
                readline.read_history_file(history_file)
            except FileNotFoundError:
                pass  # No history file yet
            
            # Configure readline
            readline.set_history_length(1000)  # Keep last 1000 commands
            
            # Save history on exit
            atexit.register(readline.write_history_file, history_file)
            
            # Enable tab completion (optional)
            readline.parse_and_bind("tab: complete")
            
            print("🔍 Command history enabled - use ↑/↓ arrow keys to navigate previous commands")
            
        except ImportError:
            print("⚠️  Command history not available (readline not installed)")
        except Exception as e:
            print(f"⚠️  Command history setup failed: {e}")
    
    def get_user_input(self, prompt: str = "🤖 Enter command: ") -> str:
        """Get user input with history support"""
        try:
            return input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            return "exit"
    
    def print_welcome(self):
        """Print welcome message and instructions"""
        print("\n" + "="*60)
        print("🏦 BOND PORTFOLIO ASSISTANT")
        print("="*60)
        print(f"🔗 API Endpoint: {self.agent.api_base_url}")
        print(f"🤖 LLM Model: GPT-4 (OpenAI)")
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
            print(f"   Notional: {format_amount_in_k(overall.portfolio_size.notional_range[0])} - {format_amount_in_k(overall.portfolio_size.notional_range[1])}")
        else:
            print(f"   Notional: Not set")
            
        if overall.portfolio_size.dirty_market_value_range:
            print(f"   Dirty MV: {format_amount_in_k(overall.portfolio_size.dirty_market_value_range[0])} - {format_amount_in_k(overall.portfolio_size.dirty_market_value_range[1])}")
        else:
            print(f"   Dirty MV: Not set")
        
        # Bond universe
        bond_universe = self.state["bond_universe"]
        print(f"\n🔍 Bond Universe ({len(bond_universe.characteristics)} characteristics):")
        if bond_universe.characteristics:
            # Group characteristics by field for better display
            characteristics_by_field = {}
            for char in bond_universe.characteristics:
                if char.field not in characteristics_by_field:
                    characteristics_by_field[char.field] = []
                characteristics_by_field[char.field].append(char)
            
            for field, chars in characteristics_by_field.items():
                if field == "sector" and len(chars) > 1:
                    # Display multiple sectors in one line
                    sector_names = [char.value for char in chars]
                    print(f"   • {field}: {', '.join(sector_names)}")
                else:
                    # Display other characteristics normally
                    for char in chars:
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
                # Get user input with history support
                user_input = self.get_user_input("\n🤖 Enter command: ")
                
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
                if self.state["current_step"] == "universe":
                    if self.state["bond_universe"].last_api_response:
                        self.show_bond_distributions()
                    else:
                        print("💡 No bond data available yet. Add some characteristics to see distributions.")
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                print("💡 Please try again with a different command.")
    
    def show_bond_distributions(self):
        """Show bond distribution charts with Rich colored bars - only for selected characteristics"""
        bond_universe = self.state["bond_universe"]
        if not bond_universe.last_api_response:
            return
        
        api_data = bond_universe.last_api_response
        aggregations = api_data.get('aggregations', {})
        
        if aggregations:
            console.print(f"\n[bold bright_blue]📊 BOND DISTRIBUTIONS[/bold bright_blue]")
            console.print(f"[bold green]Total Bonds Found: {api_data.get('total_bonds', 0)}[/bold green]")
            console.print("[dim]" + "-" * 50 + "[/dim]")
            
            # Define color schemes for different chart types
            color_schemes = {
                'sector': ['bright_red', 'bright_green', 'bright_blue', 'bright_yellow', 'bright_magenta', 'bright_cyan'],
                'rating': ['bright_green', 'green', 'yellow', 'bright_yellow', 'red', 'bright_red'],
                'yield': ['bright_blue', 'blue', 'cyan', 'bright_cyan'],
                'duration': ['bright_magenta', 'magenta', 'purple', 'bright_blue'],
                'maturity': ['bright_yellow', 'yellow', 'orange', 'red'],
                'default': ['bright_cyan', 'bright_magenta', 'bright_yellow', 'bright_green', 'bright_red', 'bright_blue']
            }
            
            # Show distributions only for characteristics that were extracted/selected
            selected_fields = set(char.field for char in bond_universe.characteristics)
            
            for field in selected_fields:
                if field in aggregations:
                    data = aggregations[field]
                    if isinstance(data, dict) and data:
                        console.print(f"\n[bold cyan]📈 {field.title()}:[/bold cyan]")
                        
                        # Get appropriate color scheme
                        colors = color_schemes.get(field, color_schemes['default'])
                        
                        # Create colorful text bar chart
                        max_count = max(data.values()) if data else 1
                        total_count = sum(data.values())
                        
                        for i, (category, count) in enumerate(data.items()):
                            bar_length = max(1, int((count / max_count) * 25))  # Longer bars for better visibility
                            percentage = (count / total_count) * 100
                            
                            # Select color from scheme (cycle through if more items than colors)
                            color = colors[i % len(colors)]
                            
                            # Create colored bar using Rich markup
                            colored_bar = f"[{color}]{'█' * bar_length}[/{color}]"
                            
                            # Format the output with Rich styling
                            console.print(f"   [white]{category:<20}[/white] {colored_bar} [bold white]{count:>4}[/bold white] [dim bright_black]({percentage:.1f}%)[/dim bright_black]")
            
            console.print("[dim]" + "-" * 50 + "[/dim]")
        else:
            console.print("\n[yellow]📊 No bond distributions available yet[/yellow]")

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
