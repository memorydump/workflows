import os
import json
import asyncio
import traceback
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import httpx
from agent import LangGraphAgent

# Load environment variables
load_dotenv()

app = FastAPI(title="Enhanced LangGraph Agent API", version="2.0.0")

# Configuration
ATLAS_API_BASE_URL = os.getenv("ATLAS_API_BASE_URL", "https://api.atlas.com")
ATLAS_API_KEY = os.getenv("ATLAS_API_KEY")
TICKERIZE_API_BASE_URL = os.getenv("TICKERIZE_API_BASE_URL", "https://api.tickerize.com")
TICKERIZE_API_KEY = os.getenv("TICKERIZE_API_KEY")
MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8080")

# Global storage for reminders (in production, use Redis or database)
active_reminders: Dict[str, Dict] = {}
# Store user context for multi-turn conversations
user_context: Dict[str, Dict] = {}

# Initialize the agent with error handling
try:
    agent = LangGraphAgent()
    print("✅ Agent initialized successfully")
except Exception as e:
    print(f"❌ Failed to initialize agent: {e}")
    traceback.print_exc()
    agent = None


class MessageRequest(BaseModel):
    message: str
    stream: bool = False
    user_id: Optional[str] = "default"  # Add user ID for context tracking


class EnhancedAgent:
    """Enhanced agent that handles tickerize, reminders, and MCP communication"""
    
    def __init__(self):
        self.atlas_service = AtlasService()
        self.tickerize_service = TickerizeService()
        self.mcp_service = MCPService()
        self.reminder_service = ReminderService()
    
    async def process_message(self, message: str, user_id: str = "default") -> str:
        """Process a message and determine the appropriate action"""
        message_lower = message.lower().strip()
        
        # Check if user is in a tickerize flow
        if user_id in user_context and user_context[user_id].get("state") == "awaiting_tickerize_selection":
            return await self._handle_tickerize_selection(message, user_id)
        
        # Check if user is awaiting client name for tickerize
        if user_id in user_context and user_context[user_id].get("state") == "awaiting_client_name":
            return await self._handle_tickerize(f"tickerize {message}", user_id)
        
        # Check for tickerize command
        if self._is_tickerize_command(message_lower):
            return await self._handle_tickerize(message, user_id)
        
        # Check for reminder command
        if self._is_reminder_command(message_lower):
            return await self._handle_reminder(message, user_id)
        
        # Check for MCP-related commands
        if self._is_mcp_command(message_lower):
            return await self._handle_mcp(message, user_id)
        
        # Check for reminder management commands
        if self._is_reminder_management(message_lower):
            return await self._handle_reminder_management(message, user_id)
        
        # Check for capabilities question
        if self._is_capabilities_question(message_lower):
            return self._get_capabilities_response()
        
        # For anything else, explain what this agent can do
        return self._get_capabilities_response()
    
    async def stream_message(self, message: str, user_id: str = "default"):
        """Stream a message response"""
        message_lower = message.lower().strip()
        
        # All commands return single response for this specialized agent
        response = await self.process_message(message, user_id)
        yield response
    
    def _is_tickerize_command(self, message: str) -> bool:
        """Check if message is a tickerize command"""
        return message.startswith("tickerize") or "tickerize" in message
    
    def _is_reminder_command(self, message: str) -> bool:
        """Check if message is a monitoring alert command"""
        alert_patterns = [
            r"alert me if",
            r"alert me when", 
            r"monitor",
            r"notify me if",
            r"notify me when",
            r"remind me when.*earnings",
            r"remind me when.*due",
            r"spread.*widens",
            r"spread.*tightens",
            r"basis points",
            r"price.*above",
            r"price.*below"
        ]
        return any(re.search(pattern, message.lower()) for pattern in alert_patterns)
    
    def _is_mcp_command(self, message: str) -> bool:
        """Check if message is a report sending command"""
        report_patterns = [
            "send.*report",
            "send.*to", 
            "factor risk report",
            "ifi report",
            "risk report",
            "performance report",
            "compliance report"
        ]
        return any(re.search(pattern, message.lower()) for pattern in report_patterns)
    
    def _is_capabilities_question(self, message: str) -> bool:
        """Check if message is asking about capabilities"""
        capability_patterns = [
            "what can you do",
            "what are your capabilities",
            "help",
            "what do you know",
            "what functions",
            "what features"
        ]
        return any(pattern in message for pattern in capability_patterns)
    
    def _get_capabilities_response(self) -> str:
        """Return information about agent capabilities"""
        return """I can help you with these 3 things:

🎯 **Tickerize Portfolio Trades**
   • Say "tickerize [client name]" to search and tickerize portfolio trades
   • Example: "tickerize ACME Corp"

📊 **Set Data Monitoring Alerts**
   • Monitor market data and set alerts for specific conditions
   • Example: "alert me if XYZ bond spread widens by 10 basis points"
   • Example: "remind me when Apple earnings are due"
   • You can also say "list alerts" to see active monitoring

📋 **Send Reports**
   • Send various reports to clients or internal teams
   • Example: "send factor risk report to Pimco"
   • Example: "send ifi report to internal team"

That's what I'm designed to do! How can I help you today?"""
    
    def _is_reminder_management(self, message: str) -> bool:
        """Check if message is about managing monitoring alerts"""
        management_patterns = [
            "list alerts",
            "show alerts",
            "active alerts", 
            "my alerts",
            "cancel alert",
            "delete alert",
            "list reminders",
            "show reminders",
            "active reminders",
            "my reminders"
        ]
        return any(pattern in message for pattern in management_patterns)
    
    async def _handle_tickerize(self, message: str, user_id: str) -> str:
        """Handle tickerize commands"""
        try:
            if message.lower().strip() == "tickerize":
                # Set context to await client name
                user_context[user_id] = {
                    "state": "awaiting_client_name"
                }
                return "Which client's portfolio trade would you like to tickerize? Please provide the client name."
            
            # Extract client name after "tickerize"
            parts = message.split()
            tickerize_index = next(i for i, word in enumerate(parts) if "tickerize" in word.lower())
            
            if tickerize_index + 1 < len(parts):
                client = " ".join(parts[tickerize_index + 1:])
            else:
                # Set context to await client name
                user_context[user_id] = {
                    "state": "awaiting_client_name"
                }
                return "Please specify which client you'd like to tickerize. Example: 'tickerize ACME Corp'"
            
            # Clear any existing context since we have the client name
            if user_id in user_context:
                del user_context[user_id]
            
            # Step 1: Call ATLAS to retrieve portfolio trades
            step1_msg = f"🔍 **Step 1:** Calling ATLAS to retrieve portfolio trades for '{client}'..."
            
            atlas_response = await self.atlas_service.search_portfolio_trades(client)
            hits = atlas_response.get("hits", [])
            
            if len(hits) == 0:
                return f"{step1_msg}\n❌ No portfolio trades found for '{client}'. Please check the client name and try again."
            
            elif len(hits) == 1:
                # Single PT inquiry - proceed directly to tickerize
                pt_inquiry = hits[0]["_source"]["screenerCentralAnalytics"]["ptInquiries"]
                sym = pt_inquiry["sym"]
                trading_name = pt_inquiry["tradingName"]
                venue = pt_inquiry["venu"]
                count = pt_inquiry["count"]
                
                step2_msg = f"✅ **Step 1 Complete:** Found 1 portfolio trade inquiry\n" \
                           f"   • Trading Name: {trading_name}\n" \
                           f"   • Symbol: {sym}\n" \
                           f"   • Venue: {venue}\n" \
                           f"   • Count: {count}\n\n" \
                           f"⚡ **Step 2:** Calling Portfolio Central to tickerize the selected portfolio trade..."
                
                tickerize_response = await self.tickerize_service.tickerize_trade(sym)
                
                return f"{step1_msg}\n{step2_msg}\n\n" \
                       f"✅ **Step 2 Complete:** Portfolio trade successfully tickerized!\n\n" \
                       f"**Tickerize Result:**\n{json.dumps(tickerize_response, indent=2)}"
            
            else:
                # Multiple PT inquiries - ask user to select
                user_context[user_id] = {
                    "state": "awaiting_tickerize_selection",
                    "client": client,
                    "options": hits
                }
                
                options_text = "\n".join([
                    f"{i+1}. Trading Name: {hit['_source']['screenerCentralAnalytics']['ptInquiries']['tradingName']} | "
                    f"Symbol: {hit['_source']['screenerCentralAnalytics']['ptInquiries']['sym']} | "
                    f"Venue: {hit['_source']['screenerCentralAnalytics']['ptInquiries']['venu']} | "
                    f"Count: {hit['_source']['screenerCentralAnalytics']['ptInquiries']['count']}"
                    for i, hit in enumerate(hits)
                ])
                
                return f"{step1_msg}\n✅ **Step 1 Complete:** Found {len(hits)} portfolio trade inquiries:\n\n" \
                       f"{options_text}\n\n" \
                       f"Please reply with the number of the trade you'd like to tickerize (1-{len(hits)})."
        
        except Exception as e:
            # Clear context on error
            if user_id in user_context:
                del user_context[user_id]
            return f"❌ Error processing tickerize request: {str(e)}"
    
    async def _handle_tickerize_selection(self, message: str, user_id: str) -> str:
        """Handle user selection for tickerize options"""
        try:
            context = user_context[user_id]
            options = context["options"]
            
            # Parse user selection
            try:
                selection = int(message.strip()) - 1
                if 0 <= selection < len(options):
                    selected_trade = options[selection]
                    pt_inquiry = selected_trade["_source"]["screenerCentralAnalytics"]["ptInquiries"]
                    sym = pt_inquiry["sym"]
                    trading_name = pt_inquiry["tradingName"]
                    venue = pt_inquiry["venu"]
                    count = pt_inquiry["count"]
                    
                    # Clear context
                    del user_context[user_id]
                    
                    step2_msg = f"⚡ **Step 2:** Calling Portfolio Central to tickerize the selected portfolio trade...\n" \
                               f"   • Selected: {trading_name} | {sym} | {venue} | Count: {count}"
                    
                    # Tickerize the selected trade
                    tickerize_response = await self.tickerize_service.tickerize_trade(sym)
                    
                    return f"{step2_msg}\n\n" \
                           f"✅ **Step 2 Complete:** Portfolio trade successfully tickerized!\n\n" \
                           f"**Tickerize Result:**\n{json.dumps(tickerize_response, indent=2)}"
                else:
                    return f"❌ Invalid selection. Please choose a number between 1 and {len(options)}."
            
            except ValueError:
                return "❌ Please reply with a valid number corresponding to your choice."
        
        except Exception as e:
            # Clear context on error
            if user_id in user_context:
                del user_context[user_id]
            return f"❌ Error processing selection: {str(e)}"
    
    async def _handle_reminder(self, message: str, user_id: str) -> str:
        """Handle reminder commands"""
        try:
            # Parse reminder request
            patterns = [
                r"remind me in (\d+) (min|mins|minutes?) (.+)",
                r"remind me in (\d+) (hour|hours?) (.+)",
                r"set a reminder for (\d+) (min|mins|minutes?) (.+)",
                r"alert me in (\d+) (min|mins|minutes?) (.+)"
            ]
            
            for pattern in patterns:
                match = re.search(pattern, message.lower())
                if match:
                    time_value = int(match.group(1))
                    time_unit = match.group(2)
                    reminder_text = match.group(3).strip()
                    
                    # Convert to minutes
                    if "hour" in time_unit:
                        minutes = time_value * 60
                    else:
                        minutes = time_value
                    
                    # Create reminder
                    reminder_id = f"reminder_{user_id}_{datetime.now().timestamp()}"
                    
                    # Start the reminder in the background
                    asyncio.create_task(self.reminder_service.set_reminder(reminder_id, reminder_text, minutes, user_id))
                    
                    trigger_time = datetime.now() + timedelta(minutes=minutes)
                    
                    return f"⏰ Reminder set successfully!\n\n" \
                           f"I'll remind you in {time_value} {time_unit} to: {reminder_text}\n" \
                           f"Trigger time: {trigger_time.strftime('%Y-%m-%d %H:%M:%S')}"
            
            return "❌ I couldn't parse your reminder request. Please use format like:\n" \
                   "• 'remind me in 10 minutes to call John'\n" \
                   "• 'remind me in 1 hour to check emails'\n" \
                   "• 'set a reminder for 30 mins to take a break'"
        
        except Exception as e:
            return f"❌ Error setting reminder: {str(e)}"
    
    async def _handle_mcp(self, message: str, user_id: str) -> str:
        """Handle report sending via MCP server"""
        try:
            message_lower = message.lower()
            
            # Parse report sending requests
            if "send" in message_lower and "report" in message_lower:
                # Extract report type and recipient
                report_match = re.search(r"send (.+?) report to (.+)", message_lower)
                if report_match:
                    report_type = report_match.group(1).strip()
                    recipient = report_match.group(2).strip()
                    
                    # Prepare MCP data
                    action = "send_report"
                    data = {
                        "report_type": report_type,
                        "recipient": recipient,
                        "user_id": user_id,
                        "timestamp": datetime.now().isoformat(),
                        "original_message": message
                    }
                    
                    # Send to MCP server
                    response = await self.mcp_service.send_message(action, data)
                    
                    return f"📋 Report sent successfully!\n\n" \
                           f"Report: {report_type} report\n" \
                           f"Recipient: {recipient}\n" \
                           f"MCP Response: {json.dumps(response, indent=2)}"
                
                else:
                    return "❌ Please specify both the report type and recipient.\n" \
                           "Example: 'send factor risk report to Pimco'"
            
            return "❌ Please specify what report you'd like to send and to whom.\n" \
                   "Examples:\n" \
                   "• 'send factor risk report to Pimco'\n" \
                   "• 'send ifi report to internal team'"
        
        except Exception as e:
            return f"❌ Error sending report via MCP server: {str(e)}"
    
    async def _handle_reminder_management(self, message: str, user_id: str) -> str:
        """Handle monitoring alert management commands"""
        try:
            message_lower = message.lower()
            
            if any(phrase in message_lower for phrase in ["list alerts", "show alerts", "active alerts", "my alerts", "list reminders", "show reminders"]):
                alerts = [alert for alert in self.reminder_service.list_active_reminders() if alert.get('user_id') == user_id]
                
                if not alerts:
                    return "📝 You have no active monitoring alerts."
                
                alert_list = []
                for alert in alerts:
                    alert_type = active_reminders.get(alert['id'], {}).get('type', 'unknown')
                    if alert_type == 'spread_monitor':
                        alert_list.append(f"📊 {alert['message']}")
                    elif alert_type == 'earnings_monitor':
                        alert_list.append(f"📈 {alert['message']}")
                    elif alert_type == 'price_monitor':
                        alert_list.append(f"💰 {alert['message']}")
                    else:
                        alert_list.append(f"📝 {alert['message']}")
                
                return f"📝 Your active monitoring alerts:\n\n" + "\n".join(alert_list)
            
            return "❌ Alert management command not recognized."
        
        except Exception as e:
            return f"❌ Error managing alerts: {str(e)}"


class AtlasService:
    """Service for interacting with ATLAS API"""
    
    def __init__(self):
        self.base_url = ATLAS_API_BASE_URL
        self.api_key = ATLAS_API_KEY
    
    async def search_portfolio_trades(self, client: str) -> Dict:
        """Search for portfolio trades by client name"""
        # Mock ATLAS response for development/testing
        import random
        
        # Simulate different scenarios based on client name for testing
        client_lower = client.lower()
        
        if "test1" in client_lower or "single" in client_lower:
            # Single result scenario
            mock_response = {
                "hits": [
                    {
                        "_source": {
                            "screenerCentralAnalytics": {
                                "ptInquiries": {
                                    "tradingName": "pimco",
                                    "sym": "PIMCO_001",
                                    "inqReceivedDateTime": 1704067200,
                                    "count": 57,
                                    "venu": "tradeWeb"
                                }
                            }
                        }
                    }
                ]
            }
        elif "test2" in client_lower or "multiple" in client_lower:
            # Multiple results scenario
            mock_response = {
                "hits": [
                    {
                        "_source": {
                            "screenerCentralAnalytics": {
                                "ptInquiries": {
                                    "tradingName": "pimco",
                                    "sym": "PIMCO_001",
                                    "inqReceivedDateTime": 1704067200,
                                    "count": 57,
                                    "venu": "tradeWeb"
                                }
                            }
                        }
                    },
                    {
                        "_source": {
                            "screenerCentralAnalytics": {
                                "ptInquiries": {
                                    "tradingName": "blackrock",
                                    "sym": "BR_002",
                                    "inqReceivedDateTime": 1704153600,
                                    "count": 23,
                                    "venu": "bloomberg"
                                }
                            }
                        }
                    },
                    {
                        "_source": {
                            "screenerCentralAnalytics": {
                                "ptInquiries": {
                                    "tradingName": "vanguard",
                                    "sym": "VG_003",
                                    "inqReceivedDateTime": 1704240000,
                                    "count": 89,
                                    "venu": "tradeWeb"
                                }
                            }
                        }
                    }
                ]
            }
        elif "empty" in client_lower or "none" in client_lower:
            # No results scenario
            mock_response = {
                "hits": []
            }
        else:
            # Random scenario for other clients
            scenarios = [
                {"hits": []},  # No results
                {  # Single result
                    "hits": [
                        {
                            "_source": {
                                "screenerCentralAnalytics": {
                                    "ptInquiries": {
                                        "tradingName": client_lower,
                                        "sym": f"{client_lower.upper()}_001",
                                        "inqReceivedDateTime": random.randint(1704000000, 1704999999),
                                        "count": random.randint(10, 100),
                                        "venu": random.choice(["tradeWeb", "bloomberg", "marketAxess"])
                                    }
                                }
                            }
                        }
                    ]
                },
                {  # Multiple results
                    "hits": [
                        {
                            "_source": {
                                "screenerCentralAnalytics": {
                                    "ptInquiries": {
                                        "tradingName": client_lower,
                                        "sym": f"{client_lower.upper()}_001",
                                        "inqReceivedDateTime": random.randint(1704000000, 1704999999),
                                        "count": random.randint(10, 100),
                                        "venu": "tradeWeb"
                                    }
                                }
                            }
                        },
                        {
                            "_source": {
                                "screenerCentralAnalytics": {
                                    "ptInquiries": {
                                        "tradingName": client_lower,
                                        "sym": f"{client_lower.upper()}_002",
                                        "inqReceivedDateTime": random.randint(1704000000, 1704999999),
                                        "count": random.randint(10, 100),
                                        "venu": "bloomberg"
                                    }
                                }
                            }
                        }
                    ]
                }
            ]
            mock_response = random.choice(scenarios)
        
        # Simulate API delay
        await asyncio.sleep(0.1)
        
        # In production, replace this with actual API call:
        # async with httpx.AsyncClient() as client_http:
        #     try:
        #         response = await client_http.get(
        #             f"{self.base_url}/portfolio-trades/search",
        #             params={"client": client},
        #             headers={"Authorization": f"Bearer {self.api_key}"}
        #         )
        #         response.raise_for_status()
        #         return response.json()
        #     except httpx.HTTPError as e:
        #         raise Exception(f"ATLAS API error: {str(e)}")
        
        return mock_response


class TickerizeService:
    """Service for interacting with Portfolio Central to tickerize trades"""
    
    def __init__(self):
        self.base_url = TICKERIZE_API_BASE_URL
        self.api_key = TICKERIZE_API_KEY
    
    async def tickerize_trade(self, sym: str) -> Dict:
        """Tickerize a specific trade using the sym identifier"""
        # Mock Portfolio Central response
        mock_response = {
            "status": "success",
            "sym": sym,
            "tickerized_data": {
                "execution_time": datetime.now().isoformat(),
                "trade_status": "processed",
                "confirmation_id": f"CONF_{sym}_{int(datetime.now().timestamp())}",
                "venue_response": "accepted"
            }
        }
        
        # Simulate API delay
        await asyncio.sleep(0.2)
        
        # In production, replace this with actual API call:
        # async with httpx.AsyncClient() as client:
        #     try:
        #         response = await client.post(
        #             f"{self.base_url}/tickerize",
        #             json={"sym": sym},
        #             headers={"Authorization": f"Bearer {self.api_key}"}
        #         )
        #         response.raise_for_status()
        #         return response.json()
        #     except httpx.HTTPError as e:
        #         raise Exception(f"Portfolio Central API error: {str(e)}")
        
        return mock_response


class MCPService:
    """Service for communicating with MCP server"""
    
    def __init__(self):
        self.server_url = MCP_SERVER_URL
    
    async def send_message(self, action: str, data: Dict[str, Any]) -> Dict:
        """Send a message to the MCP server"""
        async with httpx.AsyncClient() as client:
            try:
                response = await client.post(
                    f"{self.server_url}/mcp",
                    json={"action": action, "data": data},
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                return response.json()
            except httpx.HTTPError as e:
                raise Exception(f"MCP server error: {str(e)}")


class ReminderService:
    """Service for managing reminders and alerts"""
    
    @staticmethod
    async def set_reminder(reminder_id: str, message: str, delay_minutes: int, user_id: str = "default"):
        """Set a reminder to trigger after specified delay"""
        trigger_time = datetime.now() + timedelta(minutes=delay_minutes)
        
        active_reminders[reminder_id] = {
            "message": message,
            "trigger_time": trigger_time,
            "created_at": datetime.now(),
            "user_id": user_id
        }
        
        # Schedule the reminder
        await asyncio.sleep(delay_minutes * 60)
        
        if reminder_id in active_reminders:
            print(f"🔔 Reminder triggered for {user_id}: {message}")
            # In a real implementation, you might send this to a notification service
            # or return it through a websocket connection
            del active_reminders[reminder_id]
    
    @staticmethod
    def list_active_reminders(user_id: str = None) -> List[Dict]:
        """List all active reminders, optionally filtered by user"""
        reminders = []
        for reminder_id, data in active_reminders.items():
            if user_id is None or data.get("user_id") == user_id:
                reminders.append({
                    "id": reminder_id,
                    "message": data["message"],
                    "trigger_time": data["trigger_time"].isoformat(),
                    "minutes_remaining": max(0, int((data["trigger_time"] - datetime.now()).total_seconds() / 60)),
                    "user_id": data.get("user_id", "default")
                })
        return reminders


# Initialize enhanced agent
enhanced_agent = EnhancedAgent()


@app.get("/")
async def root():
    return {
        "message": "Enhanced LangGraph Agent API is running",
        "features": ["tickerize", "reminders", "mcp_communication"],
        "version": "2.0.0",
        "main_endpoint": "/api/agents/1"
    }


@app.get("/health")
async def health_check():
    agent_status = "healthy" if agent is not None else "agent_initialization_failed"
    return {
        "status": agent_status,
        "atlas_configured": bool(ATLAS_API_KEY),
        "tickerize_configured": bool(TICKERIZE_API_KEY),
        "mcp_server": MCP_SERVER_URL,
        "active_reminders": len(active_reminders),
        "features": {
            "tickerize": "✅ Ready",
            "reminders": "✅ Ready", 
            "mcp_communication": "✅ Ready"
        }
    }


@app.post("/api/agents/1")
async def chat_with_agent(request: MessageRequest):
    """
    Send a message to agent 1 and get a response
    Now handles tickerize, reminders, MCP communication, and regular chat
    """
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        if request.stream:
            # Return streaming response
            async def generate_stream():
                try:
                    async for chunk in enhanced_agent.stream_message(request.message, request.user_id):
                        yield chunk
                except Exception as e:
                    print(f"Streaming error: {e}")
                    traceback.print_exc()
                    yield f"Error: {str(e)}"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/plain",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive"
                }
            )
        else:
            # Non-streaming response
            response = await enhanced_agent.process_message(request.message, request.user_id)
            return PlainTextResponse(response)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Request error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.post("/api/agents/1/stream")
async def chat_with_agent_stream(request: MessageRequest):
    """
    Send a message to agent 1 and get a streaming response
    This endpoint always streams regardless of the stream parameter
    """
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        async def generate_stream():
            try:
                async for chunk in enhanced_agent.stream_message(request.message, request.user_id):
                    yield chunk
            except Exception as e:
                print(f"Streaming error: {e}")
                traceback.print_exc()
                yield f"Error: {str(e)}"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Stream request error: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@app.get("/api/agents/1/info")
async def get_agent_info():
    """
    Get information about agent 1
    """
    return {
        "agent_id": 1,
        "name": "Enhanced LangGraph Agent",
        "model": "gpt-4",
        "description": "A specialized agent that can tickerize portfolio trades, set reminders, and communicate with MCP servers",
        "streaming_enabled": True,
        "agent_initialized": agent is not None,
        "specialized_agent": True,
        "capabilities": {
            "tickerize": {
                "description": "Say 'tickerize [client name]' to search and tickerize portfolio trades",
                "examples": ["tickerize ACME Corp", "tickerize", "tickerize John's Portfolio"],
                "status": "✅ Ready" if ATLAS_API_KEY and TICKERIZE_API_KEY else "⚠️ API keys needed"
            },
            "data_monitoring": {
                "description": "Set alerts for market data monitoring",
                "examples": ["alert me if XYZ bond spread widens by 10 basis points", "remind me when Apple earnings are due"],
                "management": ["list alerts", "show my alerts"],
                "status": "✅ Ready",
                "active_count": len(active_reminders)
            },
            "report_sending": {
                "description": "Send reports to clients or internal teams",
                "examples": ["send factor risk report to Pimco", "send ifi report to internal team"],
                "server_url": MCP_SERVER_URL,
                "status": "✅ Ready"
            }
        },
        "usage": "This agent only handles tickerize, data monitoring alerts, and report sending requests. For anything else, it will explain its capabilities."
    }
