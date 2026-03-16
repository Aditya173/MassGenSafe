"""Gemini CLI native hook adapter.

This module provides the adapter for converting MassGen's hook framework
to Gemini CLI's settings.json hook format. Gemini CLI hooks are subprocess
commands that receive JSON on stdin and return JSON on stdout, configured
via `.gemini/settings.json`.

Gemini CLI hook events:
- BeforeTool: Can return {"decision": "deny", "reason": "..."} or
              {"additionalContext": "..."}
- AfterTool: Can return {"additionalContext": "..."} to append to tool result
- AfterAgent: Can return {"decision": "deny", "reason": "..."} for retry

Unlike Claude Code (in-process async functions) or Copilot (SDK callbacks),
Gemini CLI hooks use file-based IPC: the orchestrator writes a payload file
that the hook script reads when invoked by Gemini CLI as a subprocess.
"""

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ...logger_config import logger
from .base import NativeHookAdapter

if TYPE_CHECKING:
    from ..hooks import GeneralHookManager, HookType, PatternHook


# Path to the hook script that Gemini CLI will invoke as a subprocess
_HOOK_SCRIPT_PATH = Path(__file__).parent / "gemini_cli_hook_script.py"


class GeminiCLINativeHookAdapter(NativeHookAdapter):
    """Adapts MassGen hooks to Gemini CLI's settings.json hook format.

    Gemini CLI uses subprocess-based hooks configured in settings.json:
    ```json
    {
      "hooks": {
        "BeforeTool": [{
          "matcher": ".*",
          "hooks": [{"type": "command", "command": "python3 /path/to/hook.py"}]
        }],
        "AfterTool": [{
          "matcher": ".*",
          "hooks": [{"type": "command", "command": "python3 /path/to/hook.py"}]
        }]
      }
    }
    ```

    The hook script reads a payload file written by the orchestrator via
    ``write_hook_payload()``, processes it, and returns JSON on stdout.

    This adapter:
    1. Maps MassGen PRE_TOOL_USE → BeforeTool and POST_TOOL_USE → AfterTool
    2. Generates settings.json hook config pointing to the hook script
    3. Uses file-based IPC (hook_payload.json) for delivering injection content
    """

    def __init__(self, hook_dir: Path | None = None, docker_mode: bool = False):
        """Initialize the Gemini CLI native hook adapter.

        Args:
            hook_dir: Directory for hook IPC files (defaults to .gemini in cwd).
            docker_mode: When True, use ``python3`` and a workspace-local copy of
                the hook script instead of ``sys.executable`` and the host path.
                The script is copied into ``hook_dir`` by the backend before
                ``settings.json`` is written.
        """
        self._hook_dir = hook_dir
        self._hook_sequence = 0
        self.docker_mode = docker_mode

    @property
    def hook_dir(self) -> Path | None:
        """Return the hook IPC directory."""
        return self._hook_dir

    @hook_dir.setter
    def hook_dir(self, value: Path) -> None:
        self._hook_dir = value

    def supports_hook_type(self, hook_type: "HookType") -> bool:
        """Gemini CLI supports both PreToolUse (BeforeTool) and PostToolUse (AfterTool)."""
        from ..hooks import HookType as HT

        return hook_type in (HT.PRE_TOOL_USE, HT.POST_TOOL_USE)

    def convert_hook_to_native(
        self,
        hook: "PatternHook",
        hook_type: "HookType",
        context_factory: Callable[[], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Convert a MassGen hook to a Gemini CLI settings.json hook entry.

        Returns a dict representing one entry in the hooks array for the
        corresponding Gemini CLI event (BeforeTool or AfterTool).
        """
        from ..hooks import HookType as HT

        event_name = "BeforeTool" if hook_type == HT.PRE_TOOL_USE else "AfterTool"

        return {
            "event": event_name,
            "matcher": hook.matcher,
            "hook_name": hook.name,
        }

    def build_native_hooks_config(
        self,
        hook_manager: "GeneralHookManager",
        agent_id: str | None = None,
        context_factory: Callable[[], dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build Gemini CLI settings.json hooks section.

        Returns a dict suitable for merging into settings.json:
        ```json
        {
          "hooks": {
            "BeforeTool": [{"matcher": ".*", "hooks": [{"type": "command", ...}]}],
            "AfterTool": [{"matcher": ".*", "hooks": [{"type": "command", ...}]}]
          }
        }
        ```
        """
        from ..hooks import HookType as HT

        if not self._hook_dir:
            logger.warning("[GeminiCLINativeHookAdapter] No hook_dir set, cannot build hooks config")
            return {}

        hook_entries: dict[str, list[dict[str, Any]]] = {
            "BeforeTool": [],
            "AfterTool": [],
        }

        event_map = {
            HT.PRE_TOOL_USE: "BeforeTool",
            HT.POST_TOOL_USE: "AfterTool",
        }

        for hook_type, event_name in event_map.items():
            hooks = hook_manager.get_hooks_for_agent(agent_id, hook_type)
            if not hooks:
                continue

            # Collect unique matchers
            matchers: set[str] = set()
            for hook in hooks:
                matchers.add(hook.matcher)

            # Build hook command using the hook script.
            # In Docker mode the massgen source tree is not mounted inside the
            # container, so use python3 (from the container PATH) and a
            # workspace-local copy of the script (written by the backend into
            # hook_dir before settings.json is created).
            if self.docker_mode:
                python_exe = "python3"
                hook_script = str(self._hook_dir / "gemini_cli_hook_script.py")
            else:
                python_exe = sys.executable
                hook_script = str(_HOOK_SCRIPT_PATH)
            hook_dir_str = str(self._hook_dir)

            for matcher in matchers:
                hook_entries[event_name].append(
                    {
                        "matcher": matcher,
                        "hooks": [
                            {
                                "type": "command",
                                "command": (f"{python_exe} {hook_script}" f" --hook-dir {hook_dir_str}" f" --event {event_name}"),
                                "timeout": 10000,
                            },
                        ],
                    },
                )

        # Remove empty entries
        hooks_config = {k: v for k, v in hook_entries.items() if v}
        if not hooks_config:
            return {}

        return {"hooks": hooks_config}

    def merge_native_configs(
        self,
        *configs: dict[str, Any],
    ) -> dict[str, Any]:
        """Merge multiple Gemini CLI hook configs.

        Combines hook entries from multiple configs. Each config is expected
        to have a "hooks" key with BeforeTool/AfterTool arrays.
        """
        merged_hooks: dict[str, list[dict[str, Any]]] = {
            "BeforeTool": [],
            "AfterTool": [],
        }

        for config in configs:
            if not config:
                continue
            hooks = config.get("hooks", {})
            for event_name in ("BeforeTool", "AfterTool"):
                entries = hooks.get(event_name, [])
                if isinstance(entries, list):
                    merged_hooks[event_name].extend(entries)

        result_hooks = {k: v for k, v in merged_hooks.items() if v}
        if not result_hooks:
            return {}
        return {"hooks": result_hooks}

    # ── File-based IPC methods ───────────────────────────────────────────

    def write_hook_payload(
        self,
        content: str,
        event: str = "AfterTool",
        tool_matcher: str = "*",
        ttl_seconds: float = 30.0,
    ) -> None:
        """Write a hook payload file for the hook script to consume.

        The hook script reads this file when invoked by Gemini CLI and
        returns the content as additionalContext or a deny decision.

        Args:
            content: Injection text to deliver via the hook.
            event: Which hook event should consume this ("AfterTool" or "BeforeTool").
            tool_matcher: Glob pattern for which tools should receive the injection.
            ttl_seconds: Time-to-live before the payload expires.
        """
        if not self._hook_dir:
            logger.warning("[GeminiCLINativeHookAdapter] No hook_dir, cannot write payload")
            return

        self._hook_sequence += 1
        self._hook_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "inject": {"content": content, "strategy": "tool_result"},
            "event": event,
            "tool_matcher": tool_matcher,
            "expires_at": time.time() + ttl_seconds,
            "sequence": self._hook_sequence,
        }

        hook_file = self._hook_dir / "hook_payload.json"
        tmp_file = hook_file.with_suffix(".tmp")
        tmp_file.write_text(json.dumps(payload), encoding="utf-8")
        tmp_file.replace(hook_file)

        logger.info(
            "Wrote hook_payload.json (seq=%d, event=%s, %d chars)",
            self._hook_sequence,
            event,
            len(content),
        )

    def read_unconsumed_hook_content(self) -> str | None:
        """Read and remove any unconsumed hook payload.

        Called after a streaming round ends. If the hook file still exists,
        the hook script never consumed it. Returns the injection content
        so the orchestrator can carry it forward to the next round.
        """
        if not self._hook_dir:
            return None

        hook_file = self._hook_dir / "hook_payload.json"
        try:
            data = json.loads(hook_file.read_text(encoding="utf-8"))
            hook_file.unlink(missing_ok=True)
            inject = data.get("inject", {})
            content = inject.get("content")
            if content:
                logger.info(
                    "Read unconsumed hook content (%d chars) — carrying forward",
                    len(content),
                )
            return content
        except FileNotFoundError:
            return None
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed reading unconsumed hook file: %s", e)
            hook_file.unlink(missing_ok=True)
            return None

    def clear_hook_files(self) -> None:
        """Remove stale hook files. Called at start of each turn."""
        if not self._hook_dir:
            return
        for filename in ("hook_payload.json", "hook_payload.tmp"):
            (self._hook_dir / filename).unlink(missing_ok=True)
