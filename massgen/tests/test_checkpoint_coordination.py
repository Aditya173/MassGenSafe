"""
Tests for checkpoint coordination mode.

TDD: Tests written first, implementation follows.
Covers: checkpoint MCP server, proposed_actions on new_answer,
        orchestrator solo/checkpoint mode switching, gated patterns,
        and coordination tracker checkpoint events.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ============================================================================
# Phase 1: Checkpoint MCP Server
# ============================================================================


class TestCheckpointToolParameters:
    """Test checkpoint tool parameter validation."""

    def test_checkpoint_tool_requires_task(self):
        """checkpoint() must require a task parameter."""
        from massgen.mcp_tools.checkpoint._checkpoint_mcp_server import (
            validate_checkpoint_params,
        )

        with pytest.raises(ValueError, match="task"):
            validate_checkpoint_params(task="", context="", expected_actions=None)

    def test_checkpoint_tool_accepts_minimal_params(self):
        """checkpoint() with just task should be valid."""
        from massgen.mcp_tools.checkpoint._checkpoint_mcp_server import (
            validate_checkpoint_params,
        )

        result = validate_checkpoint_params(
            task="Build the auth system",
            context="",
            expected_actions=None,
        )
        assert result["task"] == "Build the auth system"

    def test_checkpoint_tool_accepts_full_params(self):
        """checkpoint() with all params should be valid."""
        from massgen.mcp_tools.checkpoint._checkpoint_mcp_server import (
            validate_checkpoint_params,
        )

        expected_actions = [
            {"tool": "mcp__vercel__deploy", "description": "Deploy to Vercel"},
        ]
        result = validate_checkpoint_params(
            task="Build and deploy",
            context="Website is ready",
            expected_actions=expected_actions,
        )
        assert result["task"] == "Build and deploy"
        assert result["context"] == "Website is ready"
        assert len(result["expected_actions"]) == 1
        assert result["expected_actions"][0]["tool"] == "mcp__vercel__deploy"

    def test_checkpoint_expected_actions_validates_tool_field(self):
        """Each expected_action must have a 'tool' field."""
        from massgen.mcp_tools.checkpoint._checkpoint_mcp_server import (
            validate_checkpoint_params,
        )

        with pytest.raises(ValueError, match="tool"):
            validate_checkpoint_params(
                task="Deploy",
                context="",
                expected_actions=[{"description": "no tool field"}],
            )


class TestCheckpointSignal:
    """Test checkpoint signal generation for orchestrator."""

    def test_build_checkpoint_signal(self):
        """Checkpoint tool should produce a signal dict for the orchestrator."""
        from massgen.mcp_tools.checkpoint._checkpoint_mcp_server import (
            build_checkpoint_signal,
        )

        signal = build_checkpoint_signal(
            task="Build the frontend",
            context="We need React",
            expected_actions=[
                {"tool": "mcp__vercel__deploy", "description": "Deploy"},
            ],
        )
        assert signal["type"] == "checkpoint"
        assert signal["task"] == "Build the frontend"
        assert signal["context"] == "We need React"
        assert len(signal["expected_actions"]) == 1

    def test_build_checkpoint_signal_minimal(self):
        """Checkpoint signal with minimal params."""
        from massgen.mcp_tools.checkpoint._checkpoint_mcp_server import (
            build_checkpoint_signal,
        )

        signal = build_checkpoint_signal(
            task="Review code",
            context="",
            expected_actions=None,
        )
        assert signal["type"] == "checkpoint"
        assert signal["task"] == "Review code"
        assert signal["context"] == ""
        assert signal["expected_actions"] == []

    def test_checkpoint_signal_written_to_file(self, tmp_path):
        """Checkpoint signal should be written to workspace for orchestrator detection."""
        from massgen.mcp_tools.checkpoint._checkpoint_mcp_server import (
            write_checkpoint_signal,
        )

        signal = {
            "type": "checkpoint",
            "task": "Build auth",
            "context": "",
            "expected_actions": [],
        }
        write_checkpoint_signal(signal, tmp_path)

        signal_file = tmp_path / ".massgen_checkpoint_signal.json"
        assert signal_file.exists()
        loaded = json.loads(signal_file.read_text())
        assert loaded["type"] == "checkpoint"
        assert loaded["task"] == "Build auth"


class TestCheckpointResult:
    """Test checkpoint result formatting."""

    def test_format_checkpoint_result(self):
        """Format checkpoint result for return to main agent."""
        from massgen.mcp_tools.checkpoint._checkpoint_mcp_server import (
            format_checkpoint_result,
        )

        result = format_checkpoint_result(
            consensus="Built the website with React",
            workspace_changes=[
                {"file": "src/App.tsx", "change": "created"},
            ],
            action_results=[
                {
                    "tool": "mcp__vercel__deploy",
                    "executed": True,
                    "result": {"url": "https://my-site.vercel.app"},
                },
            ],
        )
        assert result["consensus"] == "Built the website with React"
        assert len(result["workspace_changes"]) == 1
        assert len(result["action_results"]) == 1
        assert result["action_results"][0]["executed"] is True

    def test_format_checkpoint_result_no_actions(self):
        """Checkpoint result with no action results."""
        from massgen.mcp_tools.checkpoint._checkpoint_mcp_server import (
            format_checkpoint_result,
        )

        result = format_checkpoint_result(
            consensus="Chose PostgreSQL",
            workspace_changes=[],
            action_results=[],
        )
        assert result["consensus"] == "Chose PostgreSQL"
        assert result["workspace_changes"] == []
        assert result["action_results"] == []


# ============================================================================
# Phase 1: Subrun Utils - build_checkpoint_mcp_config
# ============================================================================


class TestBuildCheckpointMcpConfig:
    """Test checkpoint MCP config generation."""

    def test_build_checkpoint_mcp_config_basic(self):
        """Should generate MCP config for checkpoint server."""
        from massgen.mcp_tools.subrun_utils import build_checkpoint_mcp_config

        config = build_checkpoint_mcp_config(
            workspace_path=Path("/tmp/workspace"),
            agent_id="architect",
        )
        assert config["name"] == "massgen_checkpoint"
        assert "command" in config
        assert "--workspace-path" in str(config)
        assert "--agent-id" in str(config)

    def test_build_checkpoint_mcp_config_with_gated_patterns(self):
        """Should pass gated patterns to MCP config."""
        from massgen.mcp_tools.subrun_utils import build_checkpoint_mcp_config

        config = build_checkpoint_mcp_config(
            workspace_path=Path("/tmp/workspace"),
            agent_id="architect",
            gated_patterns=["mcp__vercel__deploy*"],
        )
        # Gated patterns should be encoded in args
        args_str = " ".join(config.get("args", []))
        assert "gated_patterns" in args_str or "--gated-patterns" in args_str


# ============================================================================
# Phase 2: Extended new_answer with proposed_actions
# ============================================================================


class TestNewAnswerProposedActions:
    """Test new_answer tool with proposed_actions extension."""

    def test_new_answer_default_no_proposed_actions(self):
        """Normal new_answer should NOT have proposed_actions param."""
        from massgen.tool.workflow_toolkits.new_answer import NewAnswerToolkit

        toolkit = NewAnswerToolkit()
        config = {"api_format": "chat_completions", "enable_workflow_tools": True}
        tools = toolkit.get_tools(config)
        assert len(tools) == 1
        tool_def = tools[0]

        # Get properties from the tool definition
        if "function" in tool_def:
            props = tool_def["function"]["parameters"]["properties"]
        else:
            props = tool_def["input_schema"]["properties"]

        assert "proposed_actions" not in props

    def test_new_answer_checkpoint_context_has_proposed_actions(self):
        """new_answer in checkpoint context should have proposed_actions param."""
        from massgen.tool.workflow_toolkits.new_answer import NewAnswerToolkit

        toolkit = NewAnswerToolkit()
        config = {
            "api_format": "chat_completions",
            "enable_workflow_tools": True,
            "checkpoint_context": True,
        }
        tools = toolkit.get_tools(config)
        assert len(tools) == 1
        tool_def = tools[0]

        props = tool_def["function"]["parameters"]["properties"]
        assert "proposed_actions" in props

    def test_new_answer_proposed_actions_claude_format(self):
        """proposed_actions should appear in Claude format when checkpoint context."""
        from massgen.tool.workflow_toolkits.new_answer import NewAnswerToolkit

        toolkit = NewAnswerToolkit()
        config = {
            "api_format": "claude",
            "enable_workflow_tools": True,
            "checkpoint_context": True,
        }
        tools = toolkit.get_tools(config)
        tool_def = tools[0]
        props = tool_def["input_schema"]["properties"]
        assert "proposed_actions" in props

    def test_new_answer_proposed_actions_response_format(self):
        """proposed_actions should appear in Response API format when checkpoint context."""
        from massgen.tool.workflow_toolkits.new_answer import NewAnswerToolkit

        toolkit = NewAnswerToolkit()
        config = {
            "api_format": "response",
            "enable_workflow_tools": True,
            "checkpoint_context": True,
        }
        tools = toolkit.get_tools(config)
        tool_def = tools[0]
        props = tool_def["function"]["parameters"]["properties"]
        assert "proposed_actions" in props


class TestWorkflowToolsCheckpointContext:
    """Test get_workflow_tools passes checkpoint context through."""

    def test_get_workflow_tools_with_checkpoint_context(self):
        """get_workflow_tools should pass checkpoint_context to new_answer toolkit."""
        from massgen.tool.workflow_toolkits import get_workflow_tools

        tools = get_workflow_tools(
            valid_agent_ids=["agent1", "agent2"],
            api_format="chat_completions",
            checkpoint_context=True,
        )
        # Find new_answer tool
        new_answer_tool = None
        for t in tools:
            name = t.get("name") or t.get("function", {}).get("name")
            if name == "new_answer":
                new_answer_tool = t
                break

        assert new_answer_tool is not None
        props = new_answer_tool["function"]["parameters"]["properties"]
        assert "proposed_actions" in props

    def test_get_workflow_tools_without_checkpoint_context(self):
        """get_workflow_tools without checkpoint_context should NOT have proposed_actions."""
        from massgen.tool.workflow_toolkits import get_workflow_tools

        tools = get_workflow_tools(
            valid_agent_ids=["agent1", "agent2"],
            api_format="chat_completions",
        )
        new_answer_tool = None
        for t in tools:
            name = t.get("name") or t.get("function", {}).get("name")
            if name == "new_answer":
                new_answer_tool = t
                break

        assert new_answer_tool is not None
        props = new_answer_tool["function"]["parameters"]["properties"]
        assert "proposed_actions" not in props


# ============================================================================
# Phase 4: Gated Pattern Enforcement
# ============================================================================


class TestCheckpointGatedHook:
    """Test CheckpointGatedHook for blocking gated tools."""

    def test_gated_hook_blocks_matching_tool(self):
        """Gated hook should block tools matching gated_patterns."""
        from massgen.mcp_tools.hooks import CheckpointGatedHook, HookEvent

        hook = CheckpointGatedHook(
            gated_patterns=["mcp__vercel__deploy*", "mcp__github__delete_*"],
        )
        event = HookEvent(
            hook_type="PreToolUse",
            session_id="test",
            orchestrator_id="orch",
            agent_id="agent1",
            timestamp=MagicMock(),
            tool_name="mcp__vercel__deploy_production",
            tool_input={},
        )
        result = hook(event)
        assert result.decision == "deny"
        assert "checkpoint" in result.reason.lower() or "proposed_action" in result.reason.lower()

    def test_gated_hook_allows_non_matching_tool(self):
        """Gated hook should allow tools NOT matching gated_patterns."""
        from massgen.mcp_tools.hooks import CheckpointGatedHook, HookEvent

        hook = CheckpointGatedHook(
            gated_patterns=["mcp__vercel__deploy*"],
        )
        event = HookEvent(
            hook_type="PreToolUse",
            session_id="test",
            orchestrator_id="orch",
            agent_id="agent1",
            timestamp=MagicMock(),
            tool_name="mcp__github__read_file",
            tool_input={},
        )
        result = hook(event)
        assert result.decision == "allow"

    def test_gated_hook_uses_fnmatch(self):
        """Gated patterns should use fnmatch syntax."""
        from massgen.mcp_tools.hooks import CheckpointGatedHook, HookEvent

        hook = CheckpointGatedHook(
            gated_patterns=["mcp__*__production_*"],
        )
        # Should match
        event_match = HookEvent(
            hook_type="PreToolUse",
            session_id="test",
            orchestrator_id="orch",
            agent_id="agent1",
            timestamp=MagicMock(),
            tool_name="mcp__aws__production_deploy",
            tool_input={},
        )
        result = hook(event_match)
        assert result.decision == "deny"

        # Should not match
        event_no_match = HookEvent(
            hook_type="PreToolUse",
            session_id="test",
            orchestrator_id="orch",
            agent_id="agent1",
            timestamp=MagicMock(),
            tool_name="mcp__aws__staging_deploy",
            tool_input={},
        )
        result = hook(event_no_match)
        assert result.decision == "allow"

    def test_gated_hook_empty_patterns_allows_all(self):
        """Empty gated_patterns should allow all tools."""
        from massgen.mcp_tools.hooks import CheckpointGatedHook, HookEvent

        hook = CheckpointGatedHook(gated_patterns=[])
        event = HookEvent(
            hook_type="PreToolUse",
            session_id="test",
            orchestrator_id="orch",
            agent_id="agent1",
            timestamp=MagicMock(),
            tool_name="mcp__vercel__deploy",
            tool_input={},
        )
        result = hook(event)
        assert result.decision == "allow"


# ============================================================================
# Phase 4: Coordination Tracker Checkpoint Events
# ============================================================================


class TestCoordinationTrackerCheckpointEvents:
    """Test checkpoint event types in coordination tracker."""

    def test_checkpoint_event_types_exist(self):
        """Checkpoint event types should be defined."""
        from massgen.coordination_tracker import EventType

        assert hasattr(EventType, "CHECKPOINT_CALLED")
        assert hasattr(EventType, "CHECKPOINT_AGENTS_ACTIVATED")
        assert hasattr(EventType, "CHECKPOINT_CONSENSUS_REACHED")
        assert hasattr(EventType, "CHECKPOINT_ACTION_EXECUTED")
        assert hasattr(EventType, "CHECKPOINT_ACTION_FAILED")
        assert hasattr(EventType, "CHECKPOINT_COMPLETED")

    def test_tracker_records_checkpoint_event(self):
        """Tracker should record checkpoint events."""
        from massgen.coordination_tracker import (
            CoordinationTracker,
            EventType,
        )

        tracker = CoordinationTracker()
        tracker._add_event(
            EventType.CHECKPOINT_CALLED,
            agent_id="architect",
            details="Delegating: Build the frontend",
        )
        events = [e for e in tracker.events if e.event_type == EventType.CHECKPOINT_CALLED]
        assert len(events) == 1
        assert events[0].agent_id == "architect"


# ============================================================================
# Phase 3: Config Validation
# ============================================================================


class TestCheckpointConfigValidation:
    """Test checkpoint config validation."""

    def test_valid_checkpoint_config(self):
        """Valid checkpoint config should pass validation."""
        from massgen.config_validator import ConfigValidator

        validator = ConfigValidator()
        config = {
            "agents": [
                {
                    "id": "architect",
                    "main_agent": True,
                    "backend": {"type": "claude", "model": "claude-sonnet-4-20250514"},
                },
                {
                    "id": "builder",
                    "backend": {"type": "claude", "model": "claude-sonnet-4-20250514"},
                },
            ],
            "checkpoint": {
                "enabled": True,
                "mode": "conversation",
            },
        }
        result = validator.validate_config(config)
        # Should not have errors related to checkpoint
        checkpoint_errors = [e for e in result.errors if "checkpoint" in e.message.lower() or "main_agent" in e.message.lower()]
        assert len(checkpoint_errors) == 0

    def test_multiple_main_agents_rejected(self):
        """Multiple main_agent: true should be rejected."""
        from massgen.config_validator import ConfigValidator

        validator = ConfigValidator()
        config = {
            "agents": [
                {
                    "id": "agent1",
                    "main_agent": True,
                    "backend": {"type": "claude", "model": "claude-sonnet-4-20250514"},
                },
                {
                    "id": "agent2",
                    "main_agent": True,
                    "backend": {"type": "claude", "model": "claude-sonnet-4-20250514"},
                },
            ],
        }
        result = validator.validate_config(config)
        main_agent_errors = [e for e in result.errors if "main_agent" in e.message.lower()]
        assert len(main_agent_errors) > 0

    def test_invalid_checkpoint_mode(self):
        """Invalid checkpoint mode should produce a warning or error."""
        from massgen.config_validator import ConfigValidator

        validator = ConfigValidator()
        config = {
            "agents": [
                {
                    "id": "agent1",
                    "main_agent": True,
                    "backend": {"type": "claude", "model": "claude-sonnet-4-20250514"},
                },
                {
                    "id": "agent2",
                    "backend": {"type": "claude", "model": "claude-sonnet-4-20250514"},
                },
            ],
            "checkpoint": {
                "enabled": True,
                "mode": "invalid_mode",
            },
        }
        result = validator.validate_config(config)
        mode_errors = [e for e in result.errors if "mode" in e.message.lower() and "checkpoint" in e.location.lower()]
        assert len(mode_errors) > 0


# ============================================================================
# Phase 3: Agent Config - Checkpoint Fields
# ============================================================================


class TestCheckpointAgentConfig:
    """Test checkpoint fields in CoordinationConfig."""

    def test_coordination_config_has_checkpoint_fields(self):
        """CoordinationConfig should have checkpoint-related fields."""
        from massgen.agent_config import CoordinationConfig

        config = CoordinationConfig()
        assert hasattr(config, "checkpoint_enabled")
        assert hasattr(config, "checkpoint_mode")
        assert hasattr(config, "checkpoint_guidance")
        assert hasattr(config, "checkpoint_gated_patterns")

    def test_coordination_config_checkpoint_defaults(self):
        """Checkpoint fields should have sensible defaults."""
        from massgen.agent_config import CoordinationConfig

        config = CoordinationConfig()
        assert config.checkpoint_enabled is False
        assert config.checkpoint_mode == "conversation"
        assert config.checkpoint_guidance == ""
        assert config.checkpoint_gated_patterns == []


class TestCheckpointCliParsing:
    """Test CLI parsing of checkpoint config."""

    def test_parse_coordination_config_with_checkpoint(self):
        """_parse_coordination_config should handle checkpoint fields."""
        from massgen.cli import _parse_coordination_config

        coord_cfg = {
            "checkpoint_enabled": True,
            "checkpoint_mode": "task",
            "checkpoint_guidance": "Break complex tasks into checkpoints.",
            "checkpoint_gated_patterns": ["mcp__vercel__deploy*"],
        }
        config = _parse_coordination_config(coord_cfg)
        assert config.checkpoint_enabled is True
        assert config.checkpoint_mode == "task"
        assert config.checkpoint_guidance == "Break complex tasks into checkpoints."
        assert config.checkpoint_gated_patterns == ["mcp__vercel__deploy*"]

    def test_parse_coordination_config_checkpoint_defaults(self):
        """Missing checkpoint fields should use defaults."""
        from massgen.cli import _parse_coordination_config

        coord_cfg = {}
        config = _parse_coordination_config(coord_cfg)
        assert config.checkpoint_enabled is False
        assert config.checkpoint_mode == "conversation"


# ============================================================================
# Phase 3: Backend Exclusions
# ============================================================================


class TestBackendExclusions:
    """Test that checkpoint params are excluded from API calls."""

    def test_main_agent_excluded_from_api_params(self):
        """main_agent should be in excluded params."""
        from massgen.backend.base import LLMBackend

        excluded = LLMBackend.get_base_excluded_config_params()
        assert "main_agent" in excluded

    def test_checkpoint_params_excluded(self):
        """Checkpoint-related params should be excluded from API calls."""
        from massgen.backend.base import LLMBackend

        excluded = LLMBackend.get_base_excluded_config_params()
        assert "checkpoint_enabled" in excluded
        assert "checkpoint_mode" in excluded
        assert "checkpoint_guidance" in excluded
        assert "checkpoint_gated_patterns" in excluded

    def test_api_handler_excludes_checkpoint_params(self):
        """API params handler should also exclude checkpoint params."""
        # APIParamsHandlerBase is abstract, but we can check the method exists
        # and the set contains checkpoint params via a concrete subclass
        from unittest.mock import MagicMock

        from massgen.api_params_handler._api_params_handler_base import (
            APIParamsHandlerBase,
        )

        handler = MagicMock(spec=APIParamsHandlerBase)
        handler.get_base_excluded_params = APIParamsHandlerBase.get_base_excluded_params
        excluded = handler.get_base_excluded_params(handler)
        assert "main_agent" in excluded
        assert "checkpoint_enabled" in excluded


# ============================================================================
# Phase 1: FRAMEWORK_MCPS
# ============================================================================


class TestFrameworkMcps:
    """Test that checkpoint is in FRAMEWORK_MCPS."""

    def test_checkpoint_in_framework_mcps(self):
        """massgen_checkpoint should be in FRAMEWORK_MCPS."""
        from massgen.filesystem_manager._constants import FRAMEWORK_MCPS

        assert "massgen_checkpoint" in FRAMEWORK_MCPS
