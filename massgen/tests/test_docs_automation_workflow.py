from pathlib import Path


def test_docs_workflow_stops_running_duplication_check():
    repo_root = Path(__file__).resolve().parents[2]
    workflow_path = repo_root / ".github" / "workflows" / "docs-automation.yml"
    workflow = workflow_path.read_text()

    assert "scripts/validate_links.py" in workflow
    assert "scripts/check_duplication.py" not in workflow
    assert "check-duplication" not in workflow
    assert "duplication-report" not in workflow
    assert "duplicated content" not in workflow.lower()
