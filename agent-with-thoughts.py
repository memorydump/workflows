"""
Custom Agent Component with Thoughts Output for Langflow 1.7.2

This component extends the built-in Agent component to add a second output
that emits the agent's intermediate reasoning steps (thoughts) as a Message.

INSTALLATION:
  1. In Langflow, drag an existing Agent component onto your canvas (or start fresh).
  2. Click the "Code" button on the Agent component header.
  3. Replace ALL of the code with this file's contents.
  4. Click "Check & Save".
  5. The component will now show TWO outputs:
       - "Response"  (the final agent answer — wire this to Chat Output)
       - "Agent Thoughts" (intermediate steps — wire this to a second Chat Output)

ALTERNATIVELY (custom component folder):
  1. Save this file to your custom components directory, e.g.:
       <LANGFLOW_COMPONENTS_PATH>/agents/agent_with_thoughts.py
  2. Create an __init__.py in the same folder:
       from .agent_with_thoughts import AgentWithThoughtsComponent
       __all__ = ["AgentWithThoughtsComponent"]
  3. Restart Langflow. The component appears under the "agents" category.
"""

from langchain_core.tools import StructuredTool

from langflow.base.agents.agent import LCToolsAgentComponent
from langflow.base.models.model_input_constants import (
    ALL_PROVIDER_FIELDS,
    MODEL_PROVIDERS_DICT,
)
from langflow.base.models.model_utils import get_model_name
from langflow.components.helpers import CurrentDateComponent
from langflow.components.helpers.memory import MemoryComponent
from langflow.components.langchain_utilities.tool_calling import (
    ToolCallingAgentComponent,
)
from langflow.io import BoolInput, DropdownInput, MultilineInput, Output
from langflow.schema.dotdict import dotdict
from langflow.schema.message import Message


def _set_advanced_true(component_input):
    component_input.advanced = True
    return component_input


# ---------------------------------------------------------------------------
# Model provider list — mirrors the built-in Agent component
# ---------------------------------------------------------------------------
MODEL_PROVIDERS_LIST = ["Anthropic", "Google Generative AI", "Groq", "OpenAI"]

MODELS_METADATA = {
    "Anthropic": {"icon": "Anthropic", "display_name": "Anthropic"},
    "Google Generative AI": {"icon": "GoogleGenerativeAI", "display_name": "Google"},
    "Groq": {"icon": "Groq", "display_name": "Groq"},
    "OpenAI": {"icon": "OpenAI", "display_name": "OpenAI"},
}


class AgentWithThoughtsComponent(ToolCallingAgentComponent):
    """Agent component that exposes both a final Response output AND
    an Agent Thoughts output containing intermediate reasoning steps."""

    display_name: str = "Agent with Thoughts"
    description: str = (
        "An Agent that outputs both its final response and its intermediate "
        "reasoning steps (thoughts) so you can inspect them in the Playground."
    )
    documentation: str = "https://docs.langflow.org/agents"
    icon = "bot"
    beta = False
    name = "AgentWithThoughts"

    # ------------------------------------------------------------------
    # Memory inputs (inherited from MemoryComponent, marked advanced)
    # ------------------------------------------------------------------
    memory_inputs = [
        _set_advanced_true(component_input)
        for component_input in MemoryComponent().inputs
    ]

    # ------------------------------------------------------------------
    # Filter out json_mode from OpenAI inputs
    # ------------------------------------------------------------------
    openai_inputs_filtered = [
        inp
        for inp in MODEL_PROVIDERS_DICT["OpenAI"]["inputs"]
        if not (hasattr(inp, "name") and inp.name == "json_mode")
    ]

    # ------------------------------------------------------------------
    # Inputs — identical to the stock Agent component
    # ------------------------------------------------------------------
    inputs = [
        DropdownInput(
            name="agent_llm",
            display_name="Language Model",
            info="The provider of the language model that the agent will use.",
            options=[*MODEL_PROVIDERS_LIST, "Custom"],
            value="OpenAI",
            real_time_refresh=True,
            options_metadata=[
                MODELS_METADATA[key] for key in MODEL_PROVIDERS_LIST
            ]
            + [{"icon": "brain"}],
        ),
        *openai_inputs_filtered,
        *ALL_PROVIDER_FIELDS,
        MultilineInput(
            name="system_prompt",
            display_name="Agent Instructions",
            info="System Prompt: Initial instructions and context provided to guide the agent's behavior.",
            value="You are a helpful assistant that can use tools to answer questions and perform tasks.",
            advanced=False,
        ),
        *LCToolsAgentComponent._base_inputs,
        *memory_inputs,
        BoolInput(
            name="add_current_date_tool",
            display_name="Current Date",
            advanced=True,
            info="If true, a tool that returns the current date and time will be added to the agent.",
            value=True,
        ),
    ]

    # ------------------------------------------------------------------
    # Outputs — THIS IS WHERE WE ADD THE THOUGHTS OUTPUT
    # ------------------------------------------------------------------
    outputs = [
        Output(
            name="response",
            display_name="Response",
            method="agent_response",
        ),
        Output(
            name="thoughts",
            display_name="Agent Thoughts",
            method="get_thoughts_output",
        ),
    ]

    # ------------------------------------------------------------------
    # Internal state: store thoughts captured during agent execution
    # ------------------------------------------------------------------
    _intermediate_steps_text: str = ""

    # ------------------------------------------------------------------
    # Provider / model helpers — same as stock Agent component
    # ------------------------------------------------------------------
    async def _build_llm_model(self):
        """Build the LLM based on selected provider."""
        provider = self.agent_llm
        if provider == "Custom":
            return self.get_llm()

        try:
            provider_info = MODEL_PROVIDERS_DICT.get(provider)
            if not provider_info:
                msg = f"Unknown model provider: {provider}"
                raise ValueError(msg)

            component_class = provider_info["component_class"]
            prefix = provider_info.get("prefix", "")
            inputs = provider_info.get("inputs", [])

            # Gather field values
            provider_kwargs = {}
            for field in inputs:
                field_name = field.name if hasattr(field, "name") else str(field)
                prefixed = f"{prefix}{field_name}" if prefix else field_name
                if hasattr(self, prefixed):
                    provider_kwargs[field_name] = getattr(self, prefixed)

            component_instance = component_class(**self.get_base_args())
            for key, value in provider_kwargs.items():
                setattr(component_instance, key, value)

            return await component_instance.get_llm()
        except Exception as e:
            raise ValueError(f"Error building {provider} model: {e}") from e

    def get_llm(self):
        """Retrieve the custom LLM if one is connected."""
        if hasattr(self, "agent_llm_node") and self.agent_llm_node:
            return self.agent_llm_node
        msg = "No custom LLM component connected."
        raise ValueError(msg)

    # ------------------------------------------------------------------
    # MAIN: agent_response — runs the agent, captures thoughts
    # ------------------------------------------------------------------
    async def agent_response(self) -> Message:
        """Run the agent and store intermediate steps for the thoughts output."""
        # Build the LLM
        llm_model = await self._build_llm_model()

        if not llm_model:
            msg = "No language model selected or configured."
            raise ValueError(msg)

        # Gather tools
        tools = self.tools or []

        # Add current date tool if enabled
        if getattr(self, "add_current_date_tool", True):
            try:
                current_date_tool = (
                    await CurrentDateComponent(**self.get_base_args()).to_toolkit()
                ).pop(0)
                if isinstance(current_date_tool, StructuredTool):
                    tools = [current_date_tool, *tools]
            except Exception:
                pass  # Non-critical: skip if date tool fails

        # Build the AgentExecutor via parent class
        agent_executor = self.create_agent_and_executor(llm_model, tools)

        # IMPORTANT: enable intermediate step capture
        agent_executor.return_intermediate_steps = True

        # Build the input
        input_text = self.input_value
        if hasattr(self, "system_prompt") and self.system_prompt:
            agent_executor.agent.runnable.first.messages[0].prompt.template = (
                self.system_prompt
            )

        # Invoke the agent
        result = await agent_executor.ainvoke(
            {"input": input_text},
        )

        # Extract final output
        final_output = result.get("output", "")

        # ----------------------------------------------------------
        # Format intermediate steps into readable "thoughts" text
        # ----------------------------------------------------------
        intermediate_steps = result.get("intermediate_steps", [])
        thoughts_parts = []

        for i, step in enumerate(intermediate_steps, 1):
            if len(step) >= 2:
                action, observation = step[0], step[1]
                tool_name = getattr(action, "tool", "Unknown Tool")
                tool_input = getattr(action, "tool_input", "")
                log = getattr(action, "log", "")

                thoughts_parts.append(f"--- Step {i} ---")
                if log:
                    thoughts_parts.append(f"Thought: {log.strip()}")
                thoughts_parts.append(f"Tool: {tool_name}")
                thoughts_parts.append(f"Tool Input: {tool_input}")
                thoughts_parts.append(f"Observation: {observation}")
                thoughts_parts.append("")

        if thoughts_parts:
            self._intermediate_steps_text = "\n".join(thoughts_parts)
        else:
            self._intermediate_steps_text = "(No intermediate steps captured)"

        # Return the final response
        return Message(text=final_output)

    # ------------------------------------------------------------------
    # THOUGHTS OUTPUT — emits the captured intermediate steps
    # ------------------------------------------------------------------
    async def get_thoughts_output(self) -> Message:
        """Return the agent's intermediate reasoning steps as a Message.

        NOTE: This output depends on agent_response() having run first.
        Langflow evaluates outputs based on downstream connections, so
        make sure 'Response' is also connected (or runs first).
        If thoughts are empty, it means the agent ran without tool calls.
        """
        # If agent_response hasn't been called yet, call it now
        if not self._intermediate_steps_text:
            await self.agent_response()

        return Message(text=self._intermediate_steps_text)
