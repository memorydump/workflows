"""
Agent with Thoughts Output — Langflow 1.7.2

A drop-in replacement for the built-in Agent component that adds a second
output port ("Agent Thoughts") exposing the agent's intermediate reasoning
steps (tool calls, observations, thought logs) as a Message.

HOW TO INSTALL (easiest — inline edit):
  1. Drag an Agent component onto your canvas.
  2. Click the **Code** button on the component header.
  3. Select ALL code and replace it with this file's contents.
  4. Click **Check & Save**.
  5. The component now shows two output ports:
       • Response       → wire to Chat Output (final answer)
       • Agent Thoughts → wire to a *second* Chat Output (reasoning steps)

ALTERNATIVELY (custom component folder):
  Save to <LANGFLOW_COMPONENTS_PATH>/agents/agent_with_thoughts.py
  with a matching __init__.py, then restart Langflow.
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

from langchain_core.callbacks import AsyncCallbackHandler
from pydantic import ValidationError

from lfx.components.models_and_agents.memory import MemoryComponent

if TYPE_CHECKING:
    from langchain_core.agents import AgentAction, AgentFinish
    from langchain_core.tools import Tool

from lfx.base.agents.agent import LCToolsAgentComponent
from lfx.base.agents.events import ExceptionWithMessageError
from lfx.base.models.unified_models import (
    get_language_model_options,
    get_llm,
    update_model_options_in_build_config,
)
from lfx.components.helpers import CurrentDateComponent
from lfx.components.langchain_utilities.tool_calling import ToolCallingAgentComponent
from lfx.custom.custom_component.component import get_component_toolkit
from lfx.helpers.base_model import build_model_from_schema
from lfx.inputs.inputs import BoolInput, ModelInput
from lfx.io import IntInput, MessageTextInput, MultilineInput, Output, SecretStrInput, TableInput
from lfx.log.logger import logger
from lfx.schema.data import Data
from lfx.schema.dotdict import dotdict
from lfx.schema.message import Message
from lfx.schema.table import EditMode


def set_advanced_true(component_input):
    component_input.advanced = True
    return component_input


# ---------------------------------------------------------------------------
# Custom callback handler that captures intermediate reasoning steps
# ---------------------------------------------------------------------------
class ThoughtCaptureHandler(AsyncCallbackHandler):
    """Async callback handler that records agent thoughts, tool calls,
    and observations into a list of formatted strings."""

    def __init__(self):
        super().__init__()
        self.steps: list[str] = []
        self._step_counter = 0

    async def on_agent_action(self, action: AgentAction, **kwargs: Any) -> None:
        self._step_counter += 1
        parts = [f"--- Step {self._step_counter} ---"]
        log = getattr(action, "log", "")
        if log:
            parts.append(f"Thought: {log.strip()}")
        parts.append(f"Tool: {action.tool}")
        tool_input = action.tool_input
        if isinstance(tool_input, dict):
            tool_input = json.dumps(tool_input, ensure_ascii=False)
        parts.append(f"Input: {tool_input}")
        self.steps.append("\n".join(parts))

    async def on_tool_end(self, output: str, **kwargs: Any) -> None:
        self.steps.append(f"Observation: {str(output)[:2000]}\n")

    async def on_agent_finish(self, finish: AgentFinish, **kwargs: Any) -> None:
        self.steps.append(f"--- Final Answer ---\n{finish.return_values.get('output', '')[:500]}")

    def get_formatted_thoughts(self) -> str:
        if not self.steps:
            return "(No intermediate reasoning steps captured — the agent answered directly without using tools.)"
        return "\n".join(self.steps)


# ---------------------------------------------------------------------------
# Main component
# ---------------------------------------------------------------------------
class AgentComponent(ToolCallingAgentComponent):
    display_name: str = "Agent with Thoughts"
    description: str = (
        "Define the agent's instructions, then enter a task to complete using tools. "
        "Outputs both the final response and intermediate reasoning steps."
    )
    documentation: str = "https://docs.langflow.org/agents"
    icon = "bot"
    beta = False
    name = "AgentWithThoughts"

    memory_inputs = [set_advanced_true(component_input) for component_input in MemoryComponent().inputs]

    inputs = [
        ModelInput(
            name="model",
            display_name="Language Model",
            info="Select your model provider",
            real_time_refresh=True,
            required=True,
        ),
        SecretStrInput(
            name="api_key",
            display_name="API Key",
            info="Model Provider API key",
            real_time_refresh=True,
            advanced=True,
        ),
        MultilineInput(
            name="system_prompt",
            display_name="Agent Instructions",
            info="System Prompt: Initial instructions and context provided to guide the agent's behavior.",
            value="You are a helpful assistant that can use tools to answer questions and perform tasks.",
            advanced=False,
        ),
        MessageTextInput(
            name="context_id",
            display_name="Context ID",
            info="The context ID of the chat. Adds an extra layer to the local memory.",
            value="",
            advanced=True,
        ),
        IntInput(
            name="n_messages",
            display_name="Number of Chat History Messages",
            value=100,
            info="Number of chat history messages to retrieve.",
            advanced=True,
            show=True,
        ),
        MultilineInput(
            name="format_instructions",
            display_name="Output Format Instructions",
            info="Generic Template for structured output formatting. Valid only with Structured response.",
            value=(
                "You are an AI that extracts structured JSON objects from unstructured text. "
                "Use a predefined schema with expected types (str, int, float, bool, dict). "
                "Extract ALL relevant instances that match the schema - if multiple patterns exist, capture them all. "
                "Fill missing or ambiguous values with defaults: null for missing values. "
                "Remove exact duplicates but keep variations that have different field values. "
                "Always return valid JSON in the expected format, never throw errors. "
                "If multiple objects can be extracted, return them all in the structured format."
            ),
            advanced=True,
        ),
        TableInput(
            name="output_schema",
            display_name="Output Schema",
            info=(
                "Schema Validation: Define the structure and data types for structured output. "
                "No validation if no output schema."
            ),
            advanced=True,
            required=False,
            value=[],
            table_schema=[
                {
                    "name": "name",
                    "display_name": "Name",
                    "type": "str",
                    "description": "Specify the name of the output field.",
                    "default": "field",
                    "edit_mode": EditMode.INLINE,
                },
                {
                    "name": "description",
                    "display_name": "Description",
                    "type": "str",
                    "description": "Describe the purpose of the output field.",
                    "default": "description of field",
                    "edit_mode": EditMode.POPOVER,
                },
                {
                    "name": "type",
                    "display_name": "Type",
                    "type": "str",
                    "edit_mode": EditMode.INLINE,
                    "description": ("Indicate the data type of the output field (e.g., str, int, float, bool, dict)."),
                    "options": ["str", "int", "float", "bool", "dict"],
                    "default": "str",
                },
                {
                    "name": "multiple",
                    "display_name": "As List",
                    "type": "boolean",
                    "description": "Set to True if this output field should be a list of the specified type.",
                    "default": "False",
                    "edit_mode": EditMode.INLINE,
                },
            ],
        ),
        *LCToolsAgentComponent.get_base_inputs(),
        BoolInput(
            name="add_current_date_tool",
            display_name="Current Date",
            advanced=True,
            info="If true, will add a tool to the agent that returns the current date.",
            value=True,
        ),
    ]

    # ------------------------------------------------------------------
    # TWO OUTPUTS: Response + Agent Thoughts
    # ------------------------------------------------------------------
    outputs = [
        Output(name="response", display_name="Response", method="message_response"),
        Output(name="thoughts", display_name="Agent Thoughts", method="thoughts_response"),
    ]

    # Internal storage for captured thoughts (populated by message_response)
    _thought_handler: ThoughtCaptureHandler | None = None

    # ------------------------------------------------------------------
    # get_agent_requirements — same as stock, but injects our
    # ThoughtCaptureHandler into the shared callbacks
    # ------------------------------------------------------------------
    async def get_agent_requirements(self):
        """Get the agent requirements for the agent."""
        from langchain_core.tools import StructuredTool

        llm_model = get_llm(
            model=self.model,
            user_id=self.user_id,
            api_key=self.api_key,
        )
        if llm_model is None:
            msg = "No language model selected. Please choose a model to proceed."
            raise ValueError(msg)

        # Get memory data
        self.chat_history = await self.get_memory_data()
        await logger.adebug(f"Retrieved {len(self.chat_history)} chat history messages")
        if isinstance(self.chat_history, Message):
            self.chat_history = [self.chat_history]

        # Add current date tool if enabled
        if self.add_current_date_tool:
            if not isinstance(self.tools, list):  # type: ignore[has-type]
                self.tools = []
            current_date_tool = (await CurrentDateComponent(**self.get_base_args()).to_toolkit()).pop(0)

            if not isinstance(current_date_tool, StructuredTool):
                msg = "CurrentDateComponent must be converted to a StructuredTool"
                raise TypeError(msg)
            self.tools.append(current_date_tool)

        # ---- THIS IS THE KEY CHANGE ----
        # Create a ThoughtCaptureHandler and inject it into the shared callbacks
        # so it receives on_agent_action / on_tool_end / on_agent_finish events.
        self._thought_handler = ThoughtCaptureHandler()
        shared_callbacks = self._get_shared_callbacks()
        shared_callbacks.append(self._thought_handler)
        self.set_tools_callbacks(self.tools, shared_callbacks)

        return llm_model, self.chat_history, self.tools

    # ------------------------------------------------------------------
    # message_response — runs the agent (same as stock)
    # ------------------------------------------------------------------
    async def message_response(self) -> Message:
        try:
            llm_model, self.chat_history, self.tools = await self.get_agent_requirements()
            self.set(
                llm=llm_model,
                tools=self.tools or [],
                chat_history=self.chat_history,
                input_value=self.input_value,
                system_prompt=self.system_prompt,
            )
            agent = self.create_agent_runnable()
            result = await self.run_agent(agent)

            self._agent_result = result

        except (ValueError, TypeError, KeyError) as e:
            await logger.aerror(f"{type(e).__name__}: {e!s}")
            raise
        except ExceptionWithMessageError as e:
            await logger.aerror(f"ExceptionWithMessageError occurred: {e}")
            raise
        except Exception as e:
            await logger.aerror(f"Unexpected error: {e!s}")
            raise
        else:
            return result

    # ------------------------------------------------------------------
    # thoughts_response — NEW OUTPUT: emits intermediate reasoning steps
    # ------------------------------------------------------------------
    async def thoughts_response(self) -> Message:
        """Return the agent's intermediate reasoning steps as a Message.

        This output depends on message_response() having already executed.
        If it hasn't run yet, it will be triggered automatically.
        """
        if self._thought_handler is None:
            await self.message_response()

        thoughts_text = self._thought_handler.get_formatted_thoughts()
        return Message(text=thoughts_text)

    # ------------------------------------------------------------------
    # Everything below is IDENTICAL to the stock AgentComponent
    # ------------------------------------------------------------------

    def _preprocess_schema(self, schema):
        processed_schema = []
        for field in schema:
            processed_field = {
                "name": str(field.get("name", "field")),
                "type": str(field.get("type", "str")),
                "description": str(field.get("description", "")),
                "multiple": field.get("multiple", False),
            }
            if isinstance(processed_field["multiple"], str):
                processed_field["multiple"] = processed_field["multiple"].lower() in [
                    "true",
                    "1",
                    "t",
                    "y",
                    "yes",
                ]
            processed_schema.append(processed_field)
        return processed_schema

    async def build_structured_output_base(self, content: str):
        json_pattern = r"\{.*\}"
        schema_error_msg = "Try setting an output schema"

        json_data = None
        try:
            json_data = json.loads(content)
        except json.JSONDecodeError:
            json_match = re.search(json_pattern, content, re.DOTALL)
            if json_match:
                try:
                    json_data = json.loads(json_match.group())
                except json.JSONDecodeError:
                    return {"content": content, "error": schema_error_msg}
            else:
                return {"content": content, "error": schema_error_msg}

        if not hasattr(self, "output_schema") or not self.output_schema or len(self.output_schema) == 0:
            return json_data

        try:
            processed_schema = self._preprocess_schema(self.output_schema)
            output_model = build_model_from_schema(processed_schema)

            if isinstance(json_data, list):
                validated_objects = []
                for item in json_data:
                    try:
                        validated_obj = output_model.model_validate(item)
                        validated_objects.append(validated_obj.model_dump())
                    except ValidationError as e:
                        await logger.aerror(f"Validation error for item: {e}")
                        validated_objects.append({"data": item, "validation_error": str(e)})
                return validated_objects

            try:
                validated_obj = output_model.model_validate(json_data)
                return [validated_obj.model_dump()]
            except ValidationError as e:
                await logger.aerror(f"Validation error: {e}")
                return [{"data": json_data, "validation_error": str(e)}]

        except (TypeError, ValueError) as e:
            await logger.aerror(f"Error building structured output: {e}")
            return json_data

    async def json_response(self) -> Data:
        try:
            system_components = []

            agent_instructions = getattr(self, "system_prompt", "") or ""
            if agent_instructions:
                system_components.append(f"{agent_instructions}")

            format_instructions = getattr(self, "format_instructions", "") or ""
            if format_instructions:
                system_components.append(f"Format instructions: {format_instructions}")

            if hasattr(self, "output_schema") and self.output_schema and len(self.output_schema) > 0:
                try:
                    processed_schema = self._preprocess_schema(self.output_schema)
                    output_model = build_model_from_schema(processed_schema)
                    schema_dict = output_model.model_json_schema()
                    schema_info = (
                        "You are given some text that may include format instructions, "
                        "explanations, or other content alongside a JSON schema.\n\n"
                        "Your task:\n"
                        "- Extract only the JSON schema.\n"
                        "- Return it as valid JSON.\n"
                        "- Do not include format instructions, explanations, or extra text.\n\n"
                        "Input:\n"
                        f"{json.dumps(schema_dict, indent=2)}\n\n"
                        "Output (only JSON schema):"
                    )
                    system_components.append(schema_info)
                except (ValidationError, ValueError, TypeError, KeyError) as e:
                    await logger.aerror(f"Could not build schema for prompt: {e}", exc_info=True)

            combined_instructions = "\n\n".join(system_components) if system_components else ""
            llm_model, self.chat_history, self.tools = await self.get_agent_requirements()
            self.set(
                llm=llm_model,
                tools=self.tools or [],
                chat_history=self.chat_history,
                input_value=self.input_value,
                system_prompt=combined_instructions,
            )

            try:
                structured_agent = self.create_agent_runnable()
            except (NotImplementedError, ValueError, TypeError) as e:
                await logger.aerror(f"Error with structured chat agent: {e}")
                raise
            try:
                result = await self.run_agent(structured_agent)
            except (
                ExceptionWithMessageError,
                ValueError,
                TypeError,
                RuntimeError,
            ) as e:
                await logger.aerror(f"Error with structured agent result: {e}")
                raise

            if hasattr(result, "content"):
                content = result.content
            elif hasattr(result, "text"):
                content = result.text
            else:
                content = str(result)

        except (
            ExceptionWithMessageError,
            ValueError,
            TypeError,
            NotImplementedError,
            AttributeError,
        ) as e:
            await logger.aerror(f"Error with structured chat agent: {e}")
            content_str = "No content returned from agent"
            return Data(data={"content": content_str, "error": str(e)})

        try:
            structured_output = await self.build_structured_output_base(content)

            if isinstance(structured_output, list) and structured_output:
                if len(structured_output) == 1:
                    return Data(data=structured_output[0])
                return Data(data={"results": structured_output})
            if isinstance(structured_output, dict):
                return Data(data=structured_output)
            return Data(data={"content": content})

        except (ValueError, TypeError) as e:
            await logger.aerror(f"Error in structured output processing: {e}")
            return Data(data={"content": content, "error": str(e)})

    async def get_memory_data(self):
        messages = (
            await MemoryComponent(**self.get_base_args())
            .set(
                session_id=self.graph.session_id,
                context_id=self.context_id,
                order="Ascending",
                n_messages=self.n_messages,
            )
            .retrieve_messages()
        )
        return [
            message for message in messages if getattr(message, "id", None) != getattr(self.input_value, "id", None)
        ]

    def update_input_types(self, build_config: dotdict) -> dotdict:
        for key, value in build_config.items():
            if isinstance(value, dict):
                if value.get("input_types") is None:
                    build_config[key]["input_types"] = []
            elif hasattr(value, "input_types") and value.input_types is None:
                value.input_types = []
        return build_config

    async def update_build_config(
        self,
        build_config: dotdict,
        field_value: list[dict],
        field_name: str | None = None,
    ) -> dotdict:
        def get_tool_calling_model_options(user_id=None):
            return get_language_model_options(user_id=user_id, tool_calling=True)

        build_config = update_model_options_in_build_config(
            component=self,
            build_config=dict(build_config),
            cache_key_prefix="language_model_options_tool_calling",
            get_options_func=get_tool_calling_model_options,
            field_name=field_name,
            field_value=field_value,
        )
        build_config = dotdict(build_config)

        if field_name == "model":
            self.log(str(field_value))
            build_config = self.update_input_types(build_config)

            default_keys = [
                "code",
                "_type",
                "model",
                "tools",
                "input_value",
                "add_current_date_tool",
                "system_prompt",
                "agent_description",
                "max_iterations",
                "handle_parsing_errors",
                "verbose",
            ]
            missing_keys = [key for key in default_keys if key not in build_config]
            if missing_keys:
                msg = f"Missing required keys in build_config: {missing_keys}"
                raise ValueError(msg)
        return dotdict({k: v.to_dict() if hasattr(v, "to_dict") else v for k, v in build_config.items()})

    async def _get_tools(self) -> list[Tool]:
        component_toolkit = get_component_toolkit()
        tools_names = self._build_tools_names()
        agent_description = self.get_tool_description()
        description = f"{agent_description}{tools_names}"

        tools = component_toolkit(component=self).get_tools(
            tool_name="Call_Agent",
            tool_description=description,
            callbacks=self.get_langchain_callbacks(),
        )
        if hasattr(self, "tools_metadata"):
            tools = component_toolkit(component=self, metadata=self.tools_metadata).update_tools_metadata(tools=tools)

        return tools
