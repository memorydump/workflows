import json
import re
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum
import httpx
import asyncio
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app initialization
app = FastAPI(
    title="Portfolio Trade AI Assistant",
    description="AI assistant for portfolio trade operations with tickerize functionality",
    version="1.0.0"
)

# Configuration
ATLAS_API_KEY = os.getenv("ATLAS_API_KEY", "your-atlas-api-key")
ATLAS_BASE_URL = os.getenv("ATLAS_BASE_URL", "https://atlas-api.your-domain.com")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-openai-api-key")

# Pydantic Models
class UserInput(BaseModel):
    message: str = Field(..., description="User's input message")
    session_id: Optional[str] = Field(None, description="Optional session ID for tracking")

class IntentType(str, Enum):
    TICKERIZE = "TICKERIZE"
    UNKNOWN = "UNKNOWN"

class IntentParameters(BaseModel):
    client: Optional[str] = None
    option: Optional[str] = None
    additional_keywords: Optional[List[str]] = []

class IntentResult(BaseModel):
    intent: IntentType
    confidence: float
    parameters: IntentParameters
    reasoning: str

class ValidationResult(BaseModel):
    valid: bool
    errors: List[str]
    parameters: IntentParameters

class PortfolioTrade(BaseModel):
    trade_id: str
    security_name: str
    trade_type: Optional[str] = None
    amount: Optional[str] = None
    price: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = {}

class AssistantResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Any] = None
    trades: Optional[List[PortfolioTrade]] = None
    session_id: Optional[str] = None

# Intent Classification Service
class IntentClassifier:
    def __init__(self):
        self.patterns = {
            IntentType.TICKERIZE: [
                r'\btickerize\b',
                r'\bportfolio\s+trades?\b',
                r'\bPT\s+trades?\b',
                r'\bbwic\b',
                r'\bowic\b',
                r'(show|get|find|fetch).*\b(portfolio|PT)\s+(trades?|positions?)\b'
            ]
        }
    
    async def classify_intent(self, user_input: str) -> IntentResult:
        """Classify user intent using pattern matching and LLM backup"""
        user_input_lower = user_input.lower()
        
        # Pattern-based classification
        for intent, patterns in self.patterns.items():
            for pattern in patterns:
                if re.search(pattern, user_input_lower):
                    parameters = self._extract_parameters(user_input)
                    confidence = 0.9 if parameters.client else 0.7
                    
                    return IntentResult(
                        intent=intent,
                        confidence=confidence,
                        parameters=parameters,
                        reasoning=f"Matched pattern: {pattern}"
                    )
        
        # If no pattern matches, try LLM classification (simplified version)
        return await self._llm_classify(user_input)
    
    def _extract_parameters(self, user_input: str) -> IntentParameters:
        """Extract parameters from user input"""
        client_patterns = [
            r'\b(pimco|blackrock|goldman|morgan\s+stanley|jp\s*morgan|citi|bank\s+of\s+america|wells\s+fargo)\b',
            r'tickerize\s+(\w+)',
            r'(show|get|find)\s+(\w+)\s+portfolio'
        ]
        
        option_patterns = [
            r'\b(bwic|owic)\b',
            r'\b(portfolio\s+trades?|PT\s+trades?)\b'
        ]
        
        client = None
        option = None
        additional_keywords = []
        
        # Extract client
        for pattern in client_patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                # Get the last group (client name)
                groups = match.groups()
                client = groups[-1].lower() if groups else None
                break
        
        # Extract option
        for pattern in option_patterns:
            match = re.search(pattern, user_input, re.IGNORECASE)
            if match:
                option = match.group(0).lower()
                break
        
        # Extract additional keywords
        keywords = re.findall(r'\b(trade|position|security|bond|equity)\w*\b', user_input, re.IGNORECASE)
        additional_keywords = [kw.lower() for kw in keywords]
        
        return IntentParameters(
            client=client,
            option=option,
            additional_keywords=additional_keywords
        )
    
    async def _llm_classify(self, user_input: str) -> IntentResult:
        """Fallback LLM-based classification (simplified mock)"""
        # In a real implementation, you'd call OpenAI API here
        # For now, return unknown intent
        return IntentResult(
            intent=IntentType.UNKNOWN,
            confidence=0.5,
            parameters=IntentParameters(),
            reasoning="No pattern matched, classified as unknown"
        )

# ATLAS API Service
class AtlasAPIService:
    def __init__(self):
        self.base_url = ATLAS_BASE_URL
        self.api_key = ATLAS_API_KEY
    
    async def fetch_portfolio_trades(self, parameters: IntentParameters) -> List[PortfolioTrade]:
        """Fetch portfolio trades from ATLAS system"""
        
        # Prepare query parameters
        params = {
            "action": "tickerize"
        }
        
        if parameters.client:
            params["client"] = parameters.client
        
        if parameters.option:
            params["option"] = parameters.option
        
        if parameters.additional_keywords:
            params["keywords"] = ",".join(parameters.additional_keywords)
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(
                    f"{self.base_url}/api/portfolio-trades",
                    params=params,
                    headers=headers
                )
                
                if response.status_code == 200:
                    data = response.json()
                    return self._parse_trades(data)
                elif response.status_code == 404:
                    return []
                else:
                    logger.error(f"ATLAS API error: {response.status_code} - {response.text}")
                    raise HTTPException(
                        status_code=response.status_code,
                        detail=f"ATLAS API error: {response.text}"
                    )
                    
        except httpx.TimeoutException:
            logger.error("ATLAS API timeout")
            raise HTTPException(status_code=504, detail="ATLAS API timeout")
        except Exception as e:
            logger.error(f"ATLAS API connection error: {str(e)}")
            # Return mock data for development/testing
            return self._get_mock_trades(parameters)
    
    def _parse_trades(self, data: Dict) -> List[PortfolioTrade]:
        """Parse API response into PortfolioTrade objects"""
        trades = []
        
        if isinstance(data, dict) and "trades" in data:
            trade_list = data["trades"]
        elif isinstance(data, list):
            trade_list = data
        else:
            return trades
        
        for trade_data in trade_list:
            try:
                trade = PortfolioTrade(
                    trade_id=trade_data.get("id", "N/A"),
                    security_name=trade_data.get("security", "Unknown Security"),
                    trade_type=trade_data.get("type"),
                    amount=trade_data.get("amount"),
                    price=trade_data.get("price"),
                    additional_info=trade_data.get("metadata", {})
                )
                trades.append(trade)
            except Exception as e:
                logger.warning(f"Failed to parse trade: {e}")
                continue
        
        return trades
    
    def _get_mock_trades(self, parameters: IntentParameters) -> List[PortfolioTrade]:
        """Return mock data for testing"""
        client = parameters.client or "unknown"
        
        mock_trades = [
            PortfolioTrade(
                trade_id=f"TRD_{client.upper()}_001",
                security_name=f"{client.upper()} Corporate Bond 2025",
                trade_type="BUY",
                amount="1,000,000",
                price="102.50",
                additional_info={"maturity": "2025-12-31", "rating": "AA"}
            ),
            PortfolioTrade(
                trade_id=f"TRD_{client.upper()}_002",
                security_name=f"{client.upper()} Equity Portfolio",
                trade_type="SELL",
                amount="500,000",
                price="45.75",
                additional_info={"sector": "Technology", "dividend_yield": "2.1%"}
            ),
            PortfolioTrade(
                trade_id=f"TRD_{client.upper()}_003",
                security_name=f"{client.upper()} High Yield Bond",
                trade_type="BUY",
                amount="750,000",
                price="98.25",
                additional_info={"maturity": "2027-06-15", "rating": "BB+"}
            )
        ]
        
        return mock_trades

# Validation Service
class ValidationService:
    @staticmethod
    def validate_tickerize_parameters(parameters: IntentParameters) -> ValidationResult:
        """Validate parameters for tickerize operation"""
        errors = []
        
        if not parameters.client:
            errors.append("Client name is required")
        
        if not parameters.option and not parameters.additional_keywords:
            errors.append("Option (bwic/owic) or keywords (portfolio trades, PT trades) required")
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            parameters=parameters
        )

# Response Formatter Service
class ResponseFormatter:
    @staticmethod
    def format_trade_selection(trades: List[PortfolioTrade], client: str) -> str:
        """Format trades for user selection"""
        if not trades:
            return f"No portfolio trades found for {client}. Please check the client name or try different keywords."
        
        message = f"Found {len(trades)} portfolio trades for {client}:\n\n"
        
        for i, trade in enumerate(trades, 1):
            message += f"{i}. **{trade.security_name}**\n"
            message += f"   - Trade ID: {trade.trade_id}\n"
            
            if trade.trade_type:
                message += f"   - Type: {trade.trade_type}\n"
            
            if trade.amount:
                message += f"   - Amount: ${trade.amount}\n"
            
            if trade.price:
                message += f"   - Price: ${trade.price}\n"
            
            if trade.additional_info:
                for key, value in trade.additional_info.items():
                    message += f"   - {key.replace('_', ' ').title()}: {value}\n"
            
            message += "\n"
        
        message += "Please select the trade number(s) you want to proceed with, or ask for more details about any specific trade."
        
        return message

# Main Assistant Service
class PortfolioAssistant:
    def __init__(self):
        self.intent_classifier = IntentClassifier()
        self.atlas_service = AtlasAPIService()
        self.validator = ValidationService()
        self.formatter = ResponseFormatter()
    
    async def process_request(self, user_input: str, session_id: Optional[str] = None) -> AssistantResponse:
        """Main processing pipeline"""
        try:
            # Step 1: Classify intent
            intent_result = await self.intent_classifier.classify_intent(user_input)
            logger.info(f"Classified intent: {intent_result.intent} with confidence {intent_result.confidence}")
            
            # Step 2: Route based on intent
            if intent_result.intent == IntentType.TICKERIZE:
                return await self._handle_tickerize(intent_result, session_id)
            else:
                return self._handle_unknown_intent(user_input, session_id)
                
        except Exception as e:
            logger.error(f"Error processing request: {str(e)}")
            return AssistantResponse(
                success=False,
                message="I apologize, but I'm experiencing technical difficulties. Please try again in a moment.",
                session_id=session_id
            )
    
    async def _handle_tickerize(self, intent_result: IntentResult, session_id: Optional[str]) -> AssistantResponse:
        """Handle tickerize requests"""
        
        # Validate parameters
        validation = self.validator.validate_tickerize_parameters(intent_result.parameters)
        
        if not validation.valid:
            error_message = "I need more information for the tickerize request:\n\n"
            for error in validation.errors:
                error_message += f"• {error}\n"
            
            error_message += "\nExamples:\n"
            error_message += "• 'tickerize pimco bwic'\n"
            error_message += "• 'show me blackrock portfolio trades'\n"
            error_message += "• 'get goldman PT trades'"
            
            return AssistantResponse(
                success=False,
                message=error_message,
                session_id=session_id
            )
        
        # Fetch portfolio trades
        try:
            trades = await self.atlas_service.fetch_portfolio_trades(intent_result.parameters)
            
            formatted_message = self.formatter.format_trade_selection(
                trades, 
                intent_result.parameters.client or "the client"
            )
            
            return AssistantResponse(
                success=True,
                message=formatted_message,
                trades=trades,
                data={
                    "intent": intent_result.intent,
                    "parameters": asdict(intent_result.parameters),
                    "trade_count": len(trades)
                },
                session_id=session_id
            )
            
        except HTTPException as e:
            return AssistantResponse(
                success=False,
                message=f"Error fetching portfolio trades: {e.detail}",
                session_id=session_id
            )
    
    def _handle_unknown_intent(self, user_input: str, session_id: Optional[str]) -> AssistantResponse:
        """Handle unknown intents"""
        message = """I'm a portfolio trade assistant. I can help you with:

1. **Tickerize portfolio trades** - Find and filter portfolio trades for specific clients

Examples of what I can do:
• "tickerize pimco bwic" - Get PIMCO BWIC trades
• "show me blackrock portfolio trades" - Get BlackRock portfolio trades  
• "find goldman PT trades" - Get Goldman Sachs portfolio trades

Please rephrase your request or try one of the examples above."""
        
        return AssistantResponse(
            success=True,
            message=message,
            session_id=session_id
        )

# Global assistant instance
assistant = PortfolioAssistant()

# API Routes
@app.post("/chat", response_model=AssistantResponse)
async def chat_endpoint(request: UserInput):
    """Main chat endpoint for user interactions"""
    logger.info(f"Received request: {request.message}")
    
    response = await assistant.process_request(
        user_input=request.message,
        session_id=request.session_id
    )
    
    return response

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "Portfolio Trade AI Assistant"}

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Portfolio Trade AI Assistant",
        "version": "1.0.0",
        "description": "AI assistant for portfolio trade operations",
        "endpoints": {
            "chat": "/chat - Main chat interface",
            "health": "/health - Health check",
            "docs": "/docs - API documentation"
        }
    }

# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": "An unexpected error occurred. Please try again later.",
            "error": str(exc) if os.getenv("DEBUG") == "true" else None
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
