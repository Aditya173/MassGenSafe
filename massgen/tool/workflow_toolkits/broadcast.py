"""
Broadcast toolkit for agent-to-agent and agent-to-human communication.
"""

import json
from typing import Any

from .base import BaseToolkit, ToolType


class BroadcastToolkit(BaseToolkit):
    """Broadcast communication toolkit for agent coordination."""

    def __init__(
        self,
        orchestrator: Any | None = None,
        broadcast_mode: str = "agents",
        wait_by_default: bool = True,
        sensitivity: str = "medium",
    ):
        """
        Initialize the Broadcast toolkit.

        Args:
            orchestrator: Reference to orchestrator (for accessing BroadcastChannel)
            broadcast_mode: "agents" or "human"
            wait_by_default: Default waiting behavior for ask_others()
            sensitivity: How frequently to use ask_others() ("low", "medium", "high")
        """
        self.orchestrator = orchestrator
        self.broadcast_mode = broadcast_mode
        self.wait_by_default = wait_by_default
        self.sensitivity = sensitivity

    @property
    def toolkit_id(self) -> str:
        """Unique identifier for broadcast toolkit."""
        return "broadcast"

    @property
    def toolkit_type(self) -> ToolType:
        """Type of this toolkit."""
        return ToolType.WORKFLOW

    def is_enabled(self, config: dict[str, Any]) -> bool:
        """
        Check if broadcasts are enabled in configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            True if broadcasts are enabled.
        """
        return config.get("broadcast_enabled", False)

    def set_orchestrator(self, orchestrator: Any):
        """
        Set the orchestrator reference.

        Args:
            orchestrator: Orchestrator instance
        """
        self.orchestrator = orchestrator

    def _get_sensitivity_guidance(self) -> str:
        """
        Get sensitivity-specific guidance for when to use ask_others().

        Returns:
            Guidance string based on sensitivity level
        """
        if self.sensitivity == "high":
            return "Use this frequently - whenever you're considering options, proposing approaches, or could benefit from input."
        elif self.sensitivity == "low":
            return "Use this only when blocked or for critical architectural decisions."
        else:  # medium (default)
            return "Use this for significant decisions, design choices, or when confirmation would be valuable."

    def get_tools(self, config: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Get broadcast tool definitions based on API format.

        Args:
            config: Configuration including api_format.

        Returns:
            List of broadcast tool definitions.
        """
        api_format = config.get("api_format", "chat_completions")

        tools = []

        # Get sensitivity guidance
        sensitivity_guidance = self._get_sensitivity_guidance()

        # Build description for ask_others
        target = "the human user" if self.broadcast_mode == "human" else "other agents"
        base_description = (
            f"Call this tool to ask questions to {target} for collaborative problem-solving. "
            "PREFERRED: Use the 'questions' parameter with structured questions that have predefined options - "
            "this provides a better UX and clearer responses. "
            "Only use 'question' (simple text) for truly open-ended questions where options don't make sense. "
            "IMPORTANT: Include ALL relevant context in your questions. " + sensitivity_guidance
        )

        # Define the structured question schema
        question_option_schema = {
            "type": "object",
            "properties": {
                "id": {"type": "string", "description": "Unique identifier for this option"},
                "label": {"type": "string", "description": "Display text for the option"},
                "description": {"type": "string", "description": "Optional explanation of the option"},
            },
            "required": ["id", "label"],
        }

        structured_question_schema = {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The question text to display"},
                "options": {
                    "type": "array",
                    "items": question_option_schema,
                    "description": "List of options for the human to choose from",
                },
                "multiSelect": {
                    "type": "boolean",
                    "description": "Whether multiple options can be selected (default: false)",
                },
                "allowOther": {
                    "type": "boolean",
                    "description": "Whether to allow free-form 'Other' response (default: true)",
                },
                "required": {
                    "type": "boolean",
                    "description": "Whether a response is required - cannot skip (default: false)",
                },
            },
            "required": ["text", "options"],
        }

        # Tool 1: ask_others
        if api_format == "claude":
            ask_others_tool = {
                "name": "ask_others",
                "description": base_description,
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "questions": {
                            "type": "array",
                            "items": structured_question_schema,
                            "description": (
                                "PREFERRED: Array of structured questions with predefined options. "
                                "Use this for most questions - it provides better UX and clearer responses. "
                                "Each question can have single-select or multi-select options. "
                                'Example: [{"text": "Which framework?", "options": [{"id": "react", "label": "React"}, {"id": "vue", "label": "Vue"}]}]'
                            ),
                        },
                        "question": {
                            "type": "string",
                            "description": (
                                "FALLBACK: A simple text question for truly open-ended questions where predefined options don't make sense. "
                                f"{target.capitalize()} cannot see your files or workspace, so include requirements, "
                                "constraints, and any important details they need to give a useful answer."
                            ),
                        },
                        "wait": {
                            "type": "boolean",
                            "description": (
                                f"Whether to wait for responses (default: {self.wait_by_default}). " "If true, blocks until responses collected. If false, returns request_id for polling."
                            ),
                        },
                    },
                },
            }
        else:
            # Chat completions format (OpenAI, etc.)
            # Note: strict mode doesn't work well with oneOf, so we use additionalProperties: false
            ask_others_tool = {
                "type": "function",
                "function": {
                    "name": "ask_others",
                    "description": base_description,
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "questions": {
                                "type": "array",
                                "items": structured_question_schema,
                                "description": (
                                    "PREFERRED: Array of structured questions with predefined options. "
                                    "Use this for most questions - it provides better UX and clearer responses. "
                                    "Each question can have single-select or multi-select options. "
                                    'Example: [{"text": "Which framework?", "options": [{"id": "react", "label": "React"}, {"id": "vue", "label": "Vue"}]}]'
                                ),
                            },
                            "question": {
                                "type": "string",
                                "description": (
                                    "FALLBACK: A simple text question for truly open-ended questions where predefined options don't make sense. "
                                    f"{target.capitalize()} cannot see your files or workspace, so include requirements, "
                                    "constraints, and any important details they need to give a useful answer."
                                ),
                            },
                            "wait": {
                                "type": "boolean",
                                "description": (
                                    f"Whether to wait for responses (default: {self.wait_by_default}). " "If true, blocks until responses collected. If false, returns request_id for polling."
                                ),
                            },
                        },
                    },
                },
            }

        tools.append(ask_others_tool)

        # Tool 2: respond_to_broadcast (only for agents mode, not human mode)
        if self.broadcast_mode == "agents":
            if api_format == "claude":
                respond_tool = {
                    "name": "respond_to_broadcast",
                    "description": (
                        "Submit your response to a broadcast question from another agent. "
                        "Use this tool to provide a clean, direct answer when responding to ask_others() questions. "
                        "Example: respond_to_broadcast(answer='I recommend using Hugo because it is fast and simple.')"
                    ),
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "answer": {
                                "type": "string",
                                "description": "Your complete response to the broadcast question. Be clear, concise, and directly answer what was asked.",
                            },
                        },
                        "required": ["answer"],
                    },
                }
            else:
                # Chat completions format (OpenAI, etc.)
                respond_tool = {
                    "type": "function",
                    "function": {
                        "name": "respond_to_broadcast",
                        "description": (
                            "Submit your response to a broadcast question from another agent. "
                            "Use this tool to provide a clean, direct answer when responding to ask_others() questions. "
                            "Example: respond_to_broadcast(answer='I recommend using Hugo because it is fast and simple.')"
                        ),
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "answer": {
                                    "type": "string",
                                    "description": "Your complete response to the broadcast question. Be clear, concise, and directly answer what was asked.",
                                },
                            },
                            "required": ["answer"],
                        },
                        "strict": True,
                    },
                }
            tools.append(respond_tool)

        # Tool 3: check_broadcast_status (only for polling mode)
        if not self.wait_by_default:
            if api_format == "claude":
                check_status_tool = {
                    "name": "check_broadcast_status",
                    "description": "Check the status of a broadcast request to see if responses are ready.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "request_id": {
                                "type": "string",
                                "description": "Request ID from ask_others()",
                            },
                        },
                        "required": ["request_id"],
                    },
                }
            else:
                check_status_tool = {
                    "type": "function",
                    "function": {
                        "name": "check_broadcast_status",
                        "description": "Check the status of a broadcast request to see if responses are ready.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "request_id": {
                                    "type": "string",
                                    "description": "Request ID from ask_others()",
                                },
                            },
                            "required": ["request_id"],
                        },
                    },
                }
            tools.append(check_status_tool)

        # Tool 4: get_broadcast_responses
        if api_format == "claude":
            get_responses_tool = {
                "name": "get_broadcast_responses",
                "description": "Get responses for a broadcast request (for polling mode).",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "request_id": {
                            "type": "string",
                            "description": "Request ID from ask_others()",
                        },
                    },
                    "required": ["request_id"],
                },
            }
        else:
            get_responses_tool = {
                "type": "function",
                "function": {
                    "name": "get_broadcast_responses",
                    "description": "Get responses for a broadcast request (for polling mode).",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "request_id": {
                                "type": "string",
                                "description": "Request ID from ask_others()",
                            },
                        },
                        "required": ["request_id"],
                    },
                },
            }

        if not self.wait_by_default:
            tools.append(get_responses_tool)

        return tools

    @property
    def requires_human_input(self) -> bool:
        """Check if broadcast tools require human input based on mode."""
        return self.broadcast_mode == "human"

    async def execute_ask_others(self, arguments: str, agent_id: str) -> str:
        """
        Execute ask_others tool - to be called by backend custom tool execution.

        Args:
            arguments: JSON string with question and wait parameters
            agent_id: ID of the calling agent

        Returns:
            JSON string with broadcast responses
        """
        # In human mode, serialize all ask_others calls so agents wait for each other
        # This ensures the second agent sees the first agent's Q&A before asking
        if self.broadcast_mode == "human":
            return await self._execute_ask_others_serialized(arguments, agent_id)
        else:
            return await self._execute_ask_others_impl(arguments, agent_id)

    async def _execute_ask_others_serialized(self, arguments: str, agent_id: str) -> str:
        """Execute ask_others with serialization lock for human mode."""
        from loguru import logger

        # Check if lock is already held (another agent is asking)
        if self.orchestrator.broadcast_channel._human_ask_others_lock.locked():
            logger.info(f"📢 [{agent_id}] Waiting for another agent's ask_others to complete...")

        # Acquire lock to serialize ask_others calls in human mode
        async with self.orchestrator.broadcast_channel._human_ask_others_lock:
            logger.info(f"📢 [{agent_id}] Acquired ask_others lock, proceeding with question")
            return await self._execute_ask_others_impl(arguments, agent_id)

    async def _execute_ask_others_impl(self, arguments: str, agent_id: str) -> str:
        """Core implementation of ask_others."""
        from loguru import logger

        from massgen.broadcast.broadcast_dataclasses import StructuredQuestion

        # Parse arguments
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
        wait = args.get("wait")
        if wait is None:
            wait = self.wait_by_default

        # Determine question type: structured (questions array) or simple (question string)
        questions_data = args.get("questions")
        question_str = args.get("question", "")

        if questions_data and isinstance(questions_data, list) and len(questions_data) > 0:
            # Structured questions - parse into StructuredQuestion objects
            question = [StructuredQuestion.from_dict(q) for q in questions_data]
            logger.info(f"📢 [{agent_id}] Structured ask_others with {len(question)} questions")
        else:
            # Simple string question (backward compatible)
            question = question_str
            logger.info(f"📢 [{agent_id}] Simple ask_others: {question[:50]}...")

        # In human mode, check if Q&A history already exists
        # If so, return it without prompting human again - let agent decide if they need to ask differently
        if self.broadcast_mode == "human":
            human_qa_history = self.orchestrator.broadcast_channel.get_human_qa_history()
            if human_qa_history:
                logger.info(f"📢 [{agent_id}] Q&A history exists ({len(human_qa_history)} entries), returning without prompting human again")
                return json.dumps(
                    {
                        "status": "deferred",
                        "responses": [],
                        "human_qa_history": human_qa_history,
                        "human_qa_note": (
                            "The human has already answered questions this session. "
                            "Review the history above - your question may already be answered. "
                            "If you still need different information, call ask_others with a more specific question."
                        ),
                    },
                )

        # Create and inject broadcast
        request_id = await self.orchestrator.broadcast_channel.create_broadcast(
            sender_agent_id=agent_id,
            question=question,
        )
        await self.orchestrator.broadcast_channel.inject_into_agents(request_id)

        if wait:
            # Blocking mode: wait for responses from agents and/or human
            result = await self.orchestrator.broadcast_channel.wait_for_responses(
                request_id,
                timeout=self.orchestrator.config.coordination_config.broadcast_timeout,
            )

            # Include human Q&A history in response for context (human mode only)
            # This allows agents running in parallel to see what human already answered
            response_data = {
                "status": result["status"],
                "responses": result["responses"],
            }

            if self.broadcast_mode == "human":
                human_qa_history = self.orchestrator.broadcast_channel.get_human_qa_history()
                if human_qa_history:
                    response_data["human_qa_history"] = human_qa_history
                    response_data["human_qa_note"] = "The human has answered these questions this turn. " "Check if your future questions are already covered."

            return json.dumps(response_data)
        else:
            # Polling mode: return request_id immediately
            # Also include Q&A history for context
            response_data = {
                "request_id": request_id,
                "status": "pending",
            }

            if self.broadcast_mode == "human":
                human_qa_history = self.orchestrator.broadcast_channel.get_human_qa_history()
                if human_qa_history:
                    response_data["human_qa_history"] = human_qa_history
                    response_data["human_qa_note"] = "The human has answered these questions this turn. " "Check if your future questions are already covered."

            return json.dumps(response_data)

    async def execute_check_broadcast_status(self, arguments: str, agent_id: str) -> str:
        """
        Execute check_broadcast_status tool.

        Args:
            arguments: JSON string with request_id
            agent_id: ID of the calling agent

        Returns:
            JSON string with broadcast status
        """
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
        request_id = args.get("request_id", "")

        status = self.orchestrator.broadcast_channel.get_broadcast_status(request_id)
        return json.dumps(status)

    async def execute_get_broadcast_responses(self, arguments: str, agent_id: str) -> str:
        """
        Execute get_broadcast_responses tool.

        Args:
            arguments: JSON string with request_id
            agent_id: ID of the calling agent

        Returns:
            JSON string with broadcast responses
        """
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
        request_id = args.get("request_id", "")

        responses = self.orchestrator.broadcast_channel.get_broadcast_responses(request_id)
        return json.dumps(responses)

    async def execute_respond_to_broadcast(self, arguments: str, agent_id: str) -> str:
        """
        Execute respond_to_broadcast tool (DEPRECATED - shadow agents handle responses).

        With the shadow agent architecture, broadcast responses are handled automatically
        by shadow agents that inherit the parent agent's context. This tool is kept for
        backwards compatibility but is no longer needed.

        Args:
            arguments: JSON string with answer
            agent_id: ID of the responding agent

        Returns:
            JSON string with status message
        """
        from loguru import logger

        logger.info(
            f"[{agent_id}] respond_to_broadcast called - responses are now handled " f"automatically by shadow agents",
        )

        return json.dumps(
            {
                "status": "info",
                "message": (
                    "Broadcast responses are now handled automatically by shadow agents. "
                    "You don't need to call this tool - your shadow agent has already responded. "
                    "Continue with your current work."
                ),
            },
        )
