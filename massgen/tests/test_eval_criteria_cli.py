"""Tests for the --eval-criteria CLI flag.

Verifies JSON file loading, validation, error handling,
and injection into coordination config as checklist_criteria_inline.
"""

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_argv(*extra_args: str) -> list[str]:
    """Build a minimal argv for CLI parsing with extra args appended."""
    return ["massgen", "--automation", *extra_args, "test question"]


def _parse_args(argv: list[str]):
    """Parse argv through the real CLI parser."""
    from massgen.cli import main_parser

    return main_parser().parse_args(argv[1:])


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


class TestEvalCriteriaArgParsed:
    """--eval-criteria is recognized by the argument parser."""

    def test_stored_in_args(self):
        args = _parse_args(_build_argv("--eval-criteria", "/tmp/c.json"))
        assert args.eval_criteria == "/tmp/c.json"

    def test_default_is_none(self):
        args = _parse_args(_build_argv())
        assert args.eval_criteria is None


# ---------------------------------------------------------------------------
# File loading and validation
# ---------------------------------------------------------------------------

VALID_CRITERIA = [
    {"text": "Code compiles without errors", "category": "must"},
    {"text": "Has unit tests", "category": "should"},
    {"text": "Uses type hints", "category": "could", "verify_by": "Run mypy"},
]


class TestEvalCriteriaFileLoading:
    """Valid JSON files are loaded and parsed correctly."""

    def test_valid_file_loaded(self, tmp_path: Path):
        criteria_file = tmp_path / "criteria.json"
        criteria_file.write_text(json.dumps(VALID_CRITERIA))

        from massgen.cli import _load_eval_criteria

        result = _load_eval_criteria(str(criteria_file))
        assert result == VALID_CRITERIA

    def test_preserves_verify_by_field(self, tmp_path: Path):
        criteria_file = tmp_path / "criteria.json"
        criteria_file.write_text(json.dumps(VALID_CRITERIA))

        from massgen.cli import _load_eval_criteria

        result = _load_eval_criteria(str(criteria_file))
        assert result[2]["verify_by"] == "Run mypy"


class TestEvalCriteriaErrors:
    """Invalid inputs produce clear errors."""

    def test_nonexistent_file_raises(self):
        from massgen.cli import _load_eval_criteria

        with pytest.raises(SystemExit):
            _load_eval_criteria("/nonexistent/path/criteria.json")

    def test_invalid_json_raises(self, tmp_path: Path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("not valid json {{{")

        from massgen.cli import _load_eval_criteria

        with pytest.raises(SystemExit):
            _load_eval_criteria(str(bad_file))

    def test_not_array_raises(self, tmp_path: Path):
        bad_file = tmp_path / "obj.json"
        bad_file.write_text(json.dumps({"text": "not an array"}))

        from massgen.cli import _load_eval_criteria

        with pytest.raises(SystemExit):
            _load_eval_criteria(str(bad_file))


# ---------------------------------------------------------------------------
# Injection into coordination config
# ---------------------------------------------------------------------------


class TestEvalCriteriaInjection:
    """Criteria are injected into config["orchestrator"]["coordination"]["checklist_criteria_inline"]."""

    def test_injected_into_empty_config(self):
        from massgen.cli import _inject_eval_criteria_into_config

        config: dict = {}
        _inject_eval_criteria_into_config(config, VALID_CRITERIA)

        inline = config["orchestrator"]["coordination"]["checklist_criteria_inline"]
        assert inline == VALID_CRITERIA

    def test_injected_into_existing_orchestrator(self):
        from massgen.cli import _inject_eval_criteria_into_config

        config: dict = {"orchestrator": {"snapshot_storage": "/tmp/snaps"}}
        _inject_eval_criteria_into_config(config, VALID_CRITERIA)

        inline = config["orchestrator"]["coordination"]["checklist_criteria_inline"]
        assert inline == VALID_CRITERIA
        # Existing keys preserved
        assert config["orchestrator"]["snapshot_storage"] == "/tmp/snaps"

    def test_injected_into_existing_coordination(self):
        from massgen.cli import _inject_eval_criteria_into_config

        config: dict = {
            "orchestrator": {
                "coordination": {"max_rounds": 5},
            },
        }
        _inject_eval_criteria_into_config(config, VALID_CRITERIA)

        inline = config["orchestrator"]["coordination"]["checklist_criteria_inline"]
        assert inline == VALID_CRITERIA
        # Existing coordination keys preserved
        assert config["orchestrator"]["coordination"]["max_rounds"] == 5

    def test_overrides_existing_inline_criteria(self):
        from massgen.cli import _inject_eval_criteria_into_config

        old_criteria = [{"text": "old criterion", "category": "must"}]
        config: dict = {
            "orchestrator": {
                "coordination": {"checklist_criteria_inline": old_criteria},
            },
        }
        _inject_eval_criteria_into_config(config, VALID_CRITERIA)

        inline = config["orchestrator"]["coordination"]["checklist_criteria_inline"]
        assert inline == VALID_CRITERIA
        assert inline != old_criteria
