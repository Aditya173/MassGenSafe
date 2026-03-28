"""GEPA-inspired evaluation criteria generation for MassGen.

This module generates task-specific evaluation criteria via a pre-collaboration
consensus run, replacing fixed T1-T4 items with dynamic E1-EN criteria tailored
to the actual task. When generation is disabled or fails, concrete static defaults
are used instead.

Criteria use category "primary" (at most one, the most impactful criterion),
"standard" (must-pass), or "stretch" (nice-to-have).
For backward compatibility: "must"/"core" → "standard", "should" → "standard",
"could"/"stretch" → "stretch".
"""

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from loguru import logger


@dataclass
class EvaluationCriteriaGeneratorConfig:
    """Configuration for evaluation criteria generation.

    Attributes:
        enabled: Whether criteria generation is enabled
        persist_across_turns: If True, reuse criteria across interactive turns
        min_criteria: Minimum number of criteria to generate
        max_criteria: Maximum number of criteria to generate
    """

    enabled: bool = False
    persist_across_turns: bool = False
    min_criteria: int = 4
    max_criteria: int = 7


@dataclass
class GeneratedCriterion:
    """A single evaluation criterion.

    Attributes:
        id: Criterion identifier (e.g., "E1", "E2")
        text: The criterion description text — should be an opinionated quality
            definition that takes a position on what "good" means, not just a
            dimension label. See the Anthropic harness design article for context:
            https://www.anthropic.com/engineering/harness-design-long-running-apps
        category: "primary" (THE criterion where default model behavior is weakest
            — at most one per set), "standard" (must-pass), or "stretch" (nice-to-have).
            Legacy values "must"/"core" map to "standard", "should" maps to "standard",
            "could"/"stretch" maps to "stretch".
        verify_by: Optional free-form instruction for how to gather evidence for this
            criterion. Set when reading the output text is insufficient — e.g.
            "render each slide to PNG and view visually with read_media",
            "record a video of the full animation and review the motion",
            "listen to the audio output from start to finish",
            "open in browser and test: click all links, submit forms, check states".
            None when textual inspection of the output is sufficient.
        anti_patterns: Specific failure modes that should tank the score for this
            criterion. Concrete, not abstract — e.g. "heart/fire/ocean metaphors"
            not "avoid cliches". None when not applicable.
    """

    id: str
    text: str
    category: str  # "primary", "standard", or "stretch"
    verify_by: str | None = None
    anti_patterns: list[str] | None = None


# Static defaults inspired by GEPA's diagnostic structure.
# These replace the legacy abstract T1-T4 items with concrete defaults
# that work for any task type.
_DEFAULT_CRITERIA_TEXTS = [
    ("The output directly achieves what was asked for — requirements are met," " not just approximated. Missing or partially implemented requirements" " count as failures."),
    ("No broken functionality, errors, or obvious defects. Everything that's" " present works correctly. A working output with fewer features beats a" " broken one with more."),
    ("The output is thorough — no significant gaps, thin sections, or" " placeholder content. Each component has enough depth to be genuinely" " useful, not just present."),
    ("The output shows care beyond correctness — thoughtful choices," " consistent style, attention to edge cases, or creative elements that" " distinguish it from adequate work."),
]

_DEFAULT_CATEGORIES = ["standard", "standard", "standard", "standard"]

# ---------------------------------------------------------------------------
# Domain-specific criteria presets
# ---------------------------------------------------------------------------
# Each preset maps to a list of (text, category) tuples.  The criteria are
# sourced from docs/modules/composition.md and cover the well-defined quality
# characteristics of each special primitive.

_CRITERIA_PRESETS: dict[str, list[tuple[str, str]]] = {
    "persona": [
        (
            "Each persona articulates a clear, specific perspective that would lead to"
            " meaningfully different outputs — not just surface variation in tone or"
            " vocabulary. Two personas that would produce essentially the same answer"
            " are a failure.",
            "standard",
        ),
        (
            "Personas are grounded in the actual task. Each perspective is relevant to" " the problem domain and brings a genuinely useful lens, not an arbitrary" " or forced viewpoint.",
            "standard",
        ),
        (
            "Personas are actionable instructions, not character descriptions. An agent"
            " receiving this persona knows exactly how it changes their approach,"
            " priorities, and decision-making — not just who they are pretending to be.",
            "standard",
        ),
        (
            "The persona set collectively provides coverage — the major reasonable"
            " approaches, value trade-offs, or methodological choices for this task are"
            " represented. No critical perspective is missing.",
            "standard",
        ),
        (
            "Personas are vivid enough to resist homogenization under peer pressure."
            " The perspective is strongly stated so that even after seeing other agents'"
            " answers, the core viewpoint remains distinguishable.",
            "standard",
        ),
    ],
    "decomposition": [
        (
            "Subtasks are collectively exhaustive — completing all subtasks fully"
            " produces the complete output. No significant aspect of the original task"
            " falls through the cracks between subtasks.",
            "standard",
        ),
        (
            "Subtasks have minimal coupling — each can be executed independently"
            " without requiring intermediate results from other subtasks. Where"
            " dependencies exist, they are explicit and the dependency order is"
            " specified.",
            "standard",
        ),
        (
            "Subtask scoping is balanced — no single subtask is trivial while another"
            " carries the bulk of the complexity. Work is distributed so each agent has"
            " a meaningful, roughly comparable contribution.",
            "standard",
        ),
        (
            "Each subtask description is self-contained and specific enough that an" " agent can execute it without needing to infer intent from other subtasks" " or the original prompt.",
            "standard",
        ),
        (
            "The decomposition strategy is appropriate for the task type — creative"
            " tasks split along conceptual boundaries, technical tasks along component"
            " boundaries, analytical tasks along dimension boundaries.",
            "standard",
        ),
    ],
    "evaluation": [
        (
            "Each criterion is specific to the actual task — not generic advice that" " applies to any output. A criterion that could be copy-pasted to an" " unrelated task is too vague.",
            "standard",
        ),
        (
            "Criteria are evaluable — an agent can determine pass/fail by examining the"
            ' output, not by making subjective judgments about intent. "Addresses edge'
            ' cases" is vague; "handles empty input, null values, and boundary'
            ' conditions" is evaluable.',
            "standard",
        ),
        (
            "Criteria push on dimensions where the model is weakest by default, not"
            " where it is already competent. Models already produce structurally correct,"
            " functional output by default — criteria that only check for correctness or"
            " completeness will pass on the first draft and add no iterative value."
            " At least one criterion must target a dimension where default model output"
            " is predictably mediocre: originality, distinctive voice, visual identity,"
            " architectural elegance, or domain-specific depth. The PRIMARY criterion"
            " should be the one the model needs to hear most.",
            "primary",
        ),
        (
            "Each criterion takes a position and names anti-patterns — it defines what"
            " good looks like AND what bad looks like for this specific task. A criterion"
            ' that says "uses vivid imagery" is a dimension label; one that says "uses'
            ' imagery that surprises — stock metaphors score poorly" is a quality'
            " definition. Every criterion should include concrete anti-patterns that"
            " identify how this task type typically goes wrong.",
            "standard",
        ),
        (
            "Criteria do not conflict with each other or create impossible trade-offs."
            " Meeting one criterion should not require violating another. Where genuine"
            " tensions exist, the criteria acknowledge the trade-off explicitly.",
            "standard",
        ),
    ],
    "prompt": [
        (
            "The prompt achieves its functional goal — an agent receiving this prompt"
            " would produce the intended type of output without additional"
            " clarification. Test: could you hand this to a capable model cold and get"
            " back what you need?",
            "standard",
        ),
        (
            "The prompt is appropriately scoped — it constrains enough to prevent" " unhelpful outputs but does not over-constrain in ways that eliminate" " valid approaches.",
            "standard",
        ),
        (
            "Important requirements are explicit, not implied. The prompt does not" ' depend on shared context, cultural assumptions, or "obvious" intentions' " that a model might miss.",
            "standard",
        ),
        (
            "The prompt is structured for parseability — key instructions are" " prominent, not buried in paragraphs. An agent skimming the prompt would" " still catch the critical constraints.",
            "standard",
        ),
        (
            "The prompt anticipates likely failure modes for its task type and includes"
            ' guardrails against them (e.g., "do not summarize when asked to analyze"'
            ' or "include concrete examples, not abstract principles").',
            "standard",
        ),
    ],
    "analysis": [
        (
            "The analysis identifies concrete, specific findings — not vague" " observations. Each finding points to a specific location, pattern, or" " data point in the source material.",
            "standard",
        ),
        (
            "Findings are supported by evidence from the actual data, not inferred from"
            ' assumptions about what "usually" happens. Claims include references to'
            " specific log entries, metrics, or examples.",
            "standard",
        ),
        (
            "The analysis distinguishes symptoms from root causes. Surface-level"
            ' observations (e.g., "agent 2 was slow") are traced to underlying'
            ' explanations (e.g., "agent 2 hit rate limits due to tool call volume").',
            "standard",
        ),
        (
            "Actionable recommendations follow from findings. Each significant finding" " includes a concrete suggestion for what to change, not just a description" " of what went wrong.",
            "standard",
        ),
        (
            "The analysis identifies patterns across the dataset, not just individual"
            " anomalies. Recurring behaviors, systematic biases, or structural issues"
            " are surfaced alongside one-off events.",
            "standard",
        ),
    ],
    "planning": [
        (
            "The plan captures the user's requested outcome and constraints" " without scope drift. Critical requirements are explicit, and no" " mandatory deliverable expectation is omitted.",
            "standard",
        ),
        (
            "The task graph is executable and internally consistent:" " dependencies are valid, ordering is coherent, and there are no" " contradictory or impossible steps.",
            "standard",
        ),
        (
            "Tasks describe both what to produce AND how to approach it —"
            " the method, key decisions, and constraints that guide execution."
            " 'Create the hero section' is insufficient; 'restructure the hero"
            " section: move value proposition above the fold, use existing brand"
            " palette, add a single prominent CTA' tells the executor what to"
            " actually do. Each task should be actionable without requiring the"
            " executor to infer creative or technical direction.",
            "standard",
        ),
        (
            "Each task has verification guidance matched to its type."
            " Verification may be deterministic (run tests, validate responses,"
            " check file structure) or qualitative (render to images and assess"
            " visual quality, read the output and evaluate tone, watch playback"
            " and judge pacing). Plans must NOT force numeric thresholds on"
            " inherently qualitative work — 'visually inspect the rendered page"
            " for layout balance and readability' is valid verification."
            " Verification says what to examine and what to look for.",
            "standard",
        ),
        (
            "Technology and tooling choices are explicit — frameworks,"
            " libraries, APIs, and tools are named, not left for the executor"
            " to guess. Where tasks connect or produce artifacts consumed by"
            " other tasks, interface contracts are specified: data shapes,"
            " file conventions, API signatures, or shared types.",
            "standard",
        ),
        (
            "The plan demonstrates thoughtful sequencing and risk management:"
            " chunking and prioritization reduce rework, high-risk or"
            " foundational tasks come first, and quality gates are placed"
            " where they most improve final output quality.",
            "standard",
        ),
        (
            "The plan demonstrates strategic depth — major decisions"
            " (architecture, creative direction, structure, approach) are"
            " deliberate and justified with rationale tied to the actual"
            " problem context, not just 'best practice' or 'modern trend.'"
            " Assumptions, boundaries, and trade-offs are documented with"
            " rationale rather than left implicit. If the project name could"
            " be swapped out and the plan reused unchanged, it lacks the"
            " specificity that produces excellent results.",
            "standard",
        ),
        (
            "Iterations prefer tightening existing tasks over adding new ones."
            " New tasks are justified when filling genuine gaps, but unjustified"
            " growth indicates sprawl. Descriptions, verification, and"
            " dependencies should improve in precision across rounds.",
            "standard",
        ),
        (
            "Tasks are classified as deterministic or exploratory with"
            " appropriate specification depth. Deterministic tasks (single"
            " correct path, binary verification) have exact steps and"
            " interface contracts. Exploratory tasks (multiple valid"
            " approaches, qualitative verification) have success criteria"
            " and constraints instead of implementation steps, giving the"
            " executor freedom to iterate.",
            "standard",
        ),
        (
            "The plan includes evaluation checkpoints after high-risk or"
            " exploratory chunks, and evolution hooks that flag assumptions"
            " which should trigger plan revision if invalidated during"
            " execution. The plan treats itself as a hypothesis, not a"
            " contract.",
            "standard",
        ),
    ],
    "spec": [
        (
            "Requirements are complete and unambiguous — each requirement" " describes a single, testable behavior or property. A developer" " reading the spec can implement without guessing intent.",
            "standard",
        ),
        (
            "Each requirement has concrete acceptance criteria: specific" " conditions, inputs, expected outputs, or observable behaviors" " that prove the requirement is met.",
            "standard",
        ),
        (
            "Scope boundaries are explicit — what is in scope and what is"
            " deliberately out of scope are both stated. The spec does not"
            " silently omit aspects the user would expect to be covered.",
            "standard",
        ),
        (
            "Requirements are prioritized and internally consistent — no two"
            " requirements contradict each other, and the priority or"
            " ordering reflects genuine implementation dependencies and"
            " user-facing importance.",
            "standard",
        ),
        (
            "Requirements anticipate edge cases, error states, and boundary" " conditions relevant to the domain. The spec does not only" " describe the happy path.",
            "standard",
        ),
        (
            "The spec demonstrates strategic depth — the chosen design"
            " direction, system architecture, and interaction model are"
            " deliberate and justified with rationale tied to the actual"
            " problem context and users. If the project name could be"
            " swapped out and the spec reused unchanged, it lacks the"
            " specificity that produces excellent results.",
            "standard",
        ),
        (
            "Iterations prefer tightening existing requirements over adding"
            " new ones. New requirements are justified when filling genuine"
            " gaps, but unjustified growth indicates sprawl. EARS statements,"
            " verification, and rationale should improve in precision across"
            " rounds.",
            "standard",
        ),
    ],
    "round_evaluator": [
        (
            "The evaluator packet fully follows the round_evaluator contract."
            " Required sections are present, the output stays critique-only, and"
            " it does not drift into checklist payload drafting, parent workflow"
            " tool instructions, or terminal outcome recommendations.",
            "standard",
        ),
        (
            "criteria_interpretation is rigorous for every active criterion."
            " It explains what the criterion truly demands, what excellent work"
            " would look like, and which false positives or shallow passes might"
            " otherwise slip through.",
            "standard",
        ),
        (
            "criterion_findings are evidence-grounded and criterion-specific."
            " They cite concrete details from the candidate answers, identify"
            " hidden risks, and clearly separate weak spots from source-answer"
            " strengths worth carrying forward.",
            "standard",
        ),
        (
            "cross_answer_synthesis and preserve guidance are decisive and"
            " lossless. The packet makes clear which answer is strongest on"
            " which dimension, what no answer gets right yet, and what strengths"
            " must not regress in the next revision.",
            "standard",
        ),
        (
            "improvement_spec is actionable, prioritized, and concrete enough"
            " that the parent can implement it with minimal reinterpretation."
            " It should describe what to change and how to change it, not just"
            " restate the problem.",
            "standard",
        ),
        (
            "verification_plan and evidence_gaps are specific and useful."
            " They name what still needs to be checked, what evidence is"
            " missing, and how the parent can close those gaps instead of"
            " falling back to generic 'test more' guidance.",
            "standard",
        ),
        (
            "unexplored_approaches includes at least one grounded, non-obvious" " direction that could beat every current answer rather than merely" " patching the current weaknesses.",
            "standard",
        ),
    ],
}

# Public constant for validation (used by config_validator and tests)
VALID_CRITERIA_PRESETS: frozenset[str] = frozenset(_CRITERIA_PRESETS.keys())


def criteria_from_inline(inline_list: list[dict[str, str]]) -> list[GeneratedCriterion]:
    """Convert inline criteria dicts to GeneratedCriterion objects.

    Accepts 'text' as the primary key, with 'description' and 'name' as fallbacks.

    Args:
        inline_list: List of dicts with 'text' (or 'description'/'name') and 'category' keys.

    Returns:
        List of GeneratedCriterion with E1..EN IDs.

    Raises:
        ValueError: If a criterion has no text content (no 'text', 'description', or 'name' key).
    """
    criteria: list[GeneratedCriterion] = []
    for i, item in enumerate(inline_list):
        # Accept common aliases: description, name -> text
        text = item.get("text") or item.get("description") or item.get("name")
        if not text:
            raise ValueError(
                f"Criterion {i + 1} is missing required 'text' field. " f'Expected format: {{"text": "...", "category": "primary|standard|stretch"}}. ' f"Got keys: {list(item.keys())}",
            )
        verify_by = str(item.get("verify_by") or "").strip() or None
        raw_cat = str(item.get("category", "standard")).strip().lower()
        # Map legacy category values
        if raw_cat in ("must", "core"):
            cat = "standard"
        elif raw_cat == "primary":
            cat = "primary"
        elif raw_cat in ("could", "stretch"):
            cat = "stretch"
        else:
            cat = "standard"
        raw_anti = item.get("anti_patterns")
        anti = raw_anti if isinstance(raw_anti, list) else None
        criteria.append(
            GeneratedCriterion(
                id=f"E{i + 1}",
                text=text,
                category=cat,
                verify_by=verify_by,
                anti_patterns=anti,
            ),
        )
    return criteria


def build_decomposition_execution_criteria(subtask: str) -> list[GeneratedCriterion]:
    """Build parameterized checklist criteria for executing one decomposition subtask."""
    scope = " ".join((subtask or "").split())
    if len(scope) > 140:
        scope = scope[:137].rstrip() + "..."
    if not scope:
        scope = "your assigned subtask"

    criteria = [
        (
            f"The current work substantially completes and improves the owned scope for this subtask: {scope}",
            "standard",
        ),
        (
            "Relevant peer work that touches this subtask is incorporated cleanly where needed " "(interfaces, contracts, shared assets, or adjacent integration boundaries).",
            "standard",
        ),
        (
            "Changes stay within the owned scope except for necessary adjacent integration. " "The agent does not take over unrelated work owned by other subtasks.",
            "standard",
        ),
        (
            "The current work does not introduce regressions in the owned area or shared " "contracts it depends on. Validation evidence is strong enough to support that claim.",
            "standard",
        ),
        (
            "This revision is a meaningful improvement to the owned subtask, not just churn, " "reformatting, or superficial edits.",
            "standard",
        ),
    ]
    return [GeneratedCriterion(id=f"E{i + 1}", text=text, category=category) for i, (text, category) in enumerate(criteria)]


def get_criteria_for_preset(preset: str) -> list[GeneratedCriterion]:
    """Return domain-specific criteria for a named preset.

    Args:
        preset: One of the known preset names.

    Returns:
        List of GeneratedCriterion with E1..E5 IDs.

    Raises:
        ValueError: If preset name is not recognized.
    """
    if preset not in _CRITERIA_PRESETS:
        valid = ", ".join(sorted(_CRITERIA_PRESETS.keys()))
        raise ValueError(
            f"Unknown criteria preset: '{preset}'. Valid presets: {valid}",
        )

    return [GeneratedCriterion(id=f"E{i + 1}", text=text, category=category) for i, (text, category) in enumerate(_CRITERIA_PRESETS[preset])]


# Quality/craft criterion — always appended as the last criterion.
# Ensures evaluators assess whether the output shows intentional, thoughtful
# choices beyond functional correctness. Without this, agents satisfy all
# requirements while producing output that feels like a minimum viable version.
_QUALITY_CRAFT_TEXT = (
    "The output reflects intentional, thoughtful choices — not just"
    " minimum viable execution. A knowledgeable person in this domain"
    " would recognize craft, not just correctness. The whole feels"
    " cohesive and considered, not assembled from adequate parts."
)


def get_default_criteria(has_changedoc: bool = False) -> list[GeneratedCriterion]:
    """Return static default evaluation criteria.

    These are used when generation is disabled or fails. They are concrete,
    GEPA-inspired defaults that work for any task type.

    Always appends a quality/craft criterion. The ``has_changedoc`` flag is
    retained for call-site compatibility but does not alter the fallback
    defaults.

    Args:
        has_changedoc: Retained for compatibility with existing call sites.

    Returns:
        List of GeneratedCriterion with E-prefix IDs.
    """
    criteria = [
        GeneratedCriterion(
            id=f"E{i + 1}",
            text=text,
            category=category,
        )
        for i, (text, category) in enumerate(
            zip(_DEFAULT_CRITERIA_TEXTS, _DEFAULT_CATEGORIES),
        )
    ]

    # Always append quality/craft criterion
    criteria.append(
        GeneratedCriterion(
            id=f"E{len(criteria) + 1}",
            text=_QUALITY_CRAFT_TEXT,
            category="standard",
        ),
    )

    return criteria


def _parse_criteria_response(
    response: str,
    min_criteria: int = 4,
    max_criteria: int = 7,
) -> tuple[list[GeneratedCriterion] | None, str | None]:
    """Parse LLM response into GeneratedCriterion objects.

    Tries to extract JSON from the response using multiple strategies:
    1. Direct JSON parse
    2. Extract from markdown code blocks
    3. Find JSON object by braces

    Returns (criteria, aspiration) tuple. Both may be None if parsing fails.
    """
    json_str = response.strip()

    data = _try_parse_json(json_str)

    # Strategy 2: Extract from markdown code blocks
    if data is None and "```" in json_str:
        if "```json" in json_str:
            start = json_str.find("```json") + 7
            end = json_str.find("```", start)
            if end > start:
                data = _try_parse_json(json_str[start:end].strip())
        if data is None:
            start = json_str.find("```") + 3
            end = json_str.find("```", start)
            if end > start:
                data = _try_parse_json(json_str[start:end].strip())

    # Strategy 3: Find JSON by braces
    if data is None:
        criteria_start = json_str.find('{"criteria"')
        if criteria_start >= 0:
            brace_count = 0
            json_end = -1
            for i, char in enumerate(json_str[criteria_start:]):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        json_end = criteria_start + i + 1
                        break
            if json_end > criteria_start:
                data = _try_parse_json(json_str[criteria_start:json_end])

    if data is None or "criteria" not in data:
        logger.warning("Failed to parse criteria response")
        return None, None

    try:
        aspiration = data.get("aspiration") if isinstance(data.get("aspiration"), str) else None

        raw_criteria = data["criteria"]
        if not isinstance(raw_criteria, list):
            logger.warning("criteria field is not a list")
            return None, None

        # Validate count
        if len(raw_criteria) < min_criteria:
            logger.warning(
                f"Too few criteria: {len(raw_criteria)} < {min_criteria}",
            )
            return None, None
        if len(raw_criteria) > max_criteria:
            logger.warning(
                f"Too many criteria: {len(raw_criteria)} > {max_criteria}",
            )
            return None, None

        # Parse into GeneratedCriterion objects with opinionated category values.
        criteria = []
        primary_count = 0
        for i, item in enumerate(raw_criteria):
            text = item.get("text", "")
            verify_by = item.get("verify_by") or None
            if verify_by and not isinstance(verify_by, str):
                verify_by = None
            # Extract category with legacy mapping
            raw_cat = str(item.get("category", "standard")).strip().lower()
            if raw_cat in ("must", "core"):
                cat = "standard"
            elif raw_cat == "primary":
                cat = "primary"
                primary_count += 1
            elif raw_cat in ("could", "stretch"):
                cat = "stretch"
            else:
                cat = "standard"
            # Extract anti-patterns
            raw_anti = item.get("anti_patterns")
            anti = raw_anti if isinstance(raw_anti, list) and all(isinstance(a, str) for a in raw_anti) else None
            criteria.append(
                GeneratedCriterion(
                    id=f"E{i + 1}",
                    text=text,
                    category=cat,
                    verify_by=verify_by,
                    anti_patterns=anti,
                ),
            )

        if primary_count > 1:
            logger.warning(
                f"[CriteriaParser] {primary_count} criteria marked 'primary', expected at most 1. Keeping first.",
            )
            seen_primary = False
            for c in criteria:
                if c.category == "primary":
                    if seen_primary:
                        c.category = "standard"
                    seen_primary = True

        return criteria, aspiration

    except (KeyError, TypeError, AttributeError) as e:
        logger.warning(f"Failed to extract criteria from parsed data: {e}")
        return None, None


def _try_parse_json(text: str) -> dict[str, Any] | None:
    """Attempt to parse JSON, returning None on failure."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


class EvaluationCriteriaGenerator:
    """Generates task-specific evaluation criteria via subagent coordination.

    When enabled, spawns a pre-collaboration subagent run to generate criteria
    specific to the task. Falls back to static defaults on failure.
    """

    def __init__(self):
        self.last_generation_source = "unknown"
        self.last_aspiration: str | None = None

    def _build_generation_prompt(
        self,
        task: str,
        has_changedoc: bool,
        min_criteria: int = 4,
        max_criteria: int = 7,
        has_planning_spec_context: bool = False,
    ) -> str:
        """Build the prompt for criteria generation.

        Args:
            task: The user's task description
            has_changedoc: Whether changedoc mode is active
            min_criteria: Minimum number of criteria
            max_criteria: Maximum number of criteria
            has_planning_spec_context: Whether planning/spec context is mounted
                and should be explicitly referenced by prompt guidance.

        Returns:
            The formatted prompt string
        """
        # Changedoc traceability is handled during final presentation,
        # not as an evaluation criterion.  When it was a criterion, agents
        # burned iterations just improving the changedoc instead of the
        # actual deliverable.
        changedoc_instruction = ""

        planning_context_section = ""
        if has_planning_spec_context:
            planning_context_section = """

## Planning/Spec Context Alignment
Read the mounted planning/spec context before generating criteria and align with \
it so goals, personas, and deliverable expectations stay coherent. Treat planning/spec \
files as read-only references — do not modify them.
"""

        # Prompt design informed by:
        # https://www.anthropic.com/engineering/harness-design-long-running-apps
        # Key insight: criteria shape what agents produce, not just how they're scored.
        # Opinionated criteria with anti-patterns and aspiration levels drive quality
        # leaps; generic dimension labels produce generic work.
        return f"""You are generating evaluation criteria for a multi-agent AI system.

## Task Being Evaluated
{task}
{planning_context_section}

## Your Goal
Generate {min_criteria}-{max_criteria} **opinionated** evaluation criteria that define \
what excellent work looks like for THIS task. Each criterion is not just a dimension \
to score — it is a quality principle that shapes how agents approach the work. \
A strong criterion takes a position on what "good" means and explicitly rejects \
common ways outputs go wrong.

## Aspiration Level

Before writing criteria, identify the aspiration level for this task in 1-2 phrases. \
What would genuinely excellent output look like? Not "correct and complete" — that \
is the floor. What would make someone say "this is remarkably good"? \
Examples: "publishable in a literary journal", "a senior engineer would merge this \
without changes", "a designer would screenshot this for their portfolio", "an expert \
in the field would learn something from reading this."

Your aspiration level appears in the output JSON and should inform every criterion.

## What Correctness Means

Correctness is not just "the file exists and opens." A correct output works as \
the user actually experiences it:

- **Structural correctness**: right form, can be used (file opens, code runs)
- **Content correctness**: says/computes right things (accurate, complete)
- **Experiential correctness**: behaves correctly in primary use environment \
  (text renders without overflow, visuals display as intended, interactions work)

An output that passes structural checks but fails experiential ones is a *wrong* \
output, not a mediocre one. Correctness criteria must cover all three dimensions.

Correctness is separate from **quality/craft**: a correct output can still be mediocre.

## What Makes Criteria Opinionated

A good criterion does three things:

1. **Takes a position on what "good" means** — not just "is it present?" but \
"does it achieve X quality?" with X being a specific, directional standard.

BAD (dimension label): "Uses vivid imagery."
GOOD (quality definition): "Uses imagery that surprises — that makes the reader \
see something they have seen before in a way they have not. Stock metaphors \
(heart = love, darkness = sadness) or AI-typical purple-prose descriptors \
score poorly."

BAD (dimension label): "Visual design quality."
GOOD (quality definition): "Design coherence: Does the design feel like it was \
authored by someone with a point of view, or assembled from components? Evidence \
of custom decisions — intentional spacing rhythms, a color system that creates mood, \
typography choices that reinforce hierarchy — scores highly. Unmodified component \
library defaults or generic AI aesthetics score poorly."

2. **Names specific anti-patterns to penalize** — what does bad work in this \
dimension look like? Not abstract badness, but the specific ways THIS task type \
typically goes wrong. Include these as an `anti_patterns` list in the JSON.

Examples of good anti-patterns:
- Code: "god functions, swallowed exceptions, any-typed escape hatches"
- Writing: "topic-sentence-then-three-examples structure, hedging qualifiers, \
  conclusions that summarize rather than advance"
- Design: "unmodified library defaults, centered-everything layouts, purple \
  gradients over white cards"
- Data: "cherry-picked examples, conclusions stated before evidence examined"

3. **Marks ONE criterion as "primary"** — the dimension where default model \
behavior is weakest and where improvement matters most. For creative tasks, \
this is usually originality or voice. For technical tasks, architecture or error \
handling. For design, visual distinctiveness. The primary criterion is where you \
push hardest. Set its `category` to `"primary"`.

## Requirements
1. Generate between {min_criteria} and {max_criteria} criteria
2. Each criterion must be specific to THIS task, not generic
3. Each criterion should be scoreable on a 1-10 scale with evidence
4. **Exactly ONE criterion must be `"primary"`** — the most impactful quality \
dimension for this task. All others are `"standard"` (must-pass) or `"stretch"` \
(nice-to-have).
5. **Every criterion must include `anti_patterns`** — 2-4 specific failure modes
6. **Criteria must cover distinct dimensions** — content, experience, craft, etc.
7. **For rendered/experienced artifacts**: include a dedicated rendering \
correctness criterion (no visual defects, broken interactions, etc.)
8. **Per-part quality**: include at least one criterion assessing whether EACH \
significant part independently meets a quality bar, not just the average.
{changedoc_instruction}
## Examples

For a task "Create an SVG of a pelican riding a bicycle":
- **[PRIMARY]** "Riding conviction: The composition must sell the fiction that \
this pelican is actually riding — weight distribution, contact points, and body \
angle create physical plausibility. A pelican floating above a bicycle or \
statically posed with no sense of motion fails."
  anti_patterns: ["character and vehicle as separate non-interacting elements", \
"static T-pose on seat", "missing pedal/handlebar engagement"]
- "Pelican accuracy: Immediately recognizable as a pelican from silhouette \
alone — beak with throat pouch, proportional body, correct wing structure. A \
generic bird with a long beak is not a pelican."
  anti_patterns: ["cartoon-simplified shapes that lose species identity", \
"anatomically impossible joint positions"]
- "Visual craft: The illustration evidences a considered aesthetic — not just \
accurate rendering. Color palette, line weight variation, and composition feel \
intentional."
  anti_patterns: ["flat uniform line weight", "white/empty background as default", \
"over-reliance on gradients as only visual interest"]

For a task "Write a poem about love":
- **[PRIMARY]** "Earned emotion: The poem makes the reader feel something through \
specific imagery and situation, not through stating feelings. Every emotional beat \
grounded in something concrete enough to see, hear, or touch."
  anti_patterns: ["abstract declarations ('my heart aches')", \
"greeting-card resolution", "emotional escalation without corresponding specificity"]
- "Surprise and originality: At least one moment the reader could not have predicted. \
Resistance to the gravitational pull of cliche on the subject of love."
  anti_patterns: ["heart/fire/ocean/stars as primary metaphors", \
"list-of-beautiful-things structure", "ending that restates the opening sentiment"]

Criteria name a quality axis with an opinion — they do NOT prescribe specific \
quantities, thresholds, or implementation choices.

BAD (prescriptive): "The website contains at least 4 pages"
GOOD (evaluative): "Topic coverage: all major aspects addressed with meaningful depth"

BAD (whole-output only): "The output shows intentional design choices"
GOOD (per-part): "Per-section quality: each significant section independently \
demonstrates craft — no section is carried by the strength of others. Evaluate the \
weakest section, not the average."

## Output Format
Return JSON with this structure:
{{
    "aspiration": "[1-2 phrase quality ceiling for this task]",
    "criteria": [
        {{
            "text": "[Aspect]: [opinionated quality definition].",
            "category": "primary",
            "anti_patterns": ["specific failure mode 1", "specific failure mode 2"],
            "verify_by": "evidence gathering instructions if needed"
        }},
        {{
            "text": "[Aspect]: [opinionated quality definition].",
            "category": "standard",
            "anti_patterns": ["failure mode 1", "failure mode 2"]
        }}
    ]
}}

**`verify_by` field**: Required whenever the criterion involves experiential correctness \
or craft that cannot be assessed by reading the source alone. Describe WHAT EVIDENCE to \
gather and WHAT TO CHECK — not which specific application or GUI to use. The evaluator \
will choose the best available tool (rendering, screenshots, browser automation, code \
execution, computer use, etc.) based on their capabilities.

State the full scope (all pages, all slides, full playback — not a sample) and list \
the specific defects or properties to look for.

- Rendered output (slides, pages, images): render ALL pages/slides to images and inspect \
  each for specific defects (e.g. text overflow, clipped elements, unreadable font sizes \
  below Npt, element collisions, blank content areas)
- Interactive output (web apps, forms): test all navigation links, form submissions, \
  button actions, and interactive state changes — list what each interaction should do
- Motion/animation: capture and review full animation playback — list expected motion \
  behavior and timing
- Audio/video: listen to or watch the complete output — list what to assess (clarity, \
  pacing, content accuracy)
- Executable code: run with representative inputs and check outputs against expected results

Do NOT name specific desktop applications (e.g. "open in PowerPoint", "view in Finder"). \
Do NOT describe GUI-specific actions (e.g. "hover to see cursor change", "right-click and \
select"). Instead describe the observable property to verify and let the evaluator choose \
the method.

Omit only when the criterion can be fully assessed by reading the output text or \
inspecting the source file structure.

Write the JSON to a file called `criteria.json` in your workspace.
Generate evaluation criteria now for the task above."""

    async def generate_criteria_via_subagent(
        self,
        task: str,
        agent_configs: list[dict[str, Any]],
        has_changedoc: bool,
        parent_workspace: str,
        log_directory: str | None,
        orchestrator_id: str,
        min_criteria: int = 4,
        max_criteria: int = 7,
        on_subagent_started: Callable | None = None,
        voting_sensitivity: str | None = None,
        voting_threshold: int | None = None,
        has_planning_spec_context: bool = False,
    ) -> list[GeneratedCriterion]:
        """Generate criteria via a subagent run.

        Args:
            task: The user's task
            agent_configs: Parent agent configs to inherit models from
            has_changedoc: Whether changedoc mode is active
            parent_workspace: Path to parent workspace
            log_directory: Path to log directory
            orchestrator_id: Parent orchestrator ID
            min_criteria: Minimum criteria count
            max_criteria: Maximum criteria count
            on_subagent_started: Callback when subagent starts
            voting_sensitivity: Optional voting sensitivity to pass through to
                the pre-collaboration subagent coordination config.
            voting_threshold: Optional voting threshold to pass through to
                the pre-collaboration subagent coordination config.
            has_planning_spec_context: Whether planning/spec context is mounted
                and should be explicitly referenced by prompt guidance.

        Returns:
            List of GeneratedCriterion objects
        """
        logger.info("Generating evaluation criteria via subagent")

        # Build workspace
        criteria_workspace = os.path.join(parent_workspace, ".criteria_generation")
        try:
            os.makedirs(criteria_workspace, exist_ok=True)
            context_md = os.path.join(criteria_workspace, "CONTEXT.md")
            with open(context_md, "w", encoding="utf-8") as f:
                f.write(
                    "# Evaluation Criteria Generation\n\n" f"Task:\n{task}\n\n" "Goal: Generate task-specific evaluation criteria in criteria.json.\n",
                )
        except Exception as e:
            logger.warning(f"Failed to prepare criteria workspace: {e}")
            criteria_workspace = parent_workspace

        try:
            from massgen.subagent.manager import SubagentManager
            from massgen.subagent.models import SubagentOrchestratorConfig

            # Simplified agent configs (no tools, pure LLM reasoning)
            simplified = []
            for i, config in enumerate(agent_configs):
                backend = config.get("backend", {})
                backend_cfg: dict = {
                    "type": backend.get("type", "openai"),
                    "model": backend.get("model"),
                    "enable_mcp_command_line": False,
                    "enable_code_based_tools": False,
                    # Without command-line MCP execution, keep file-operation MCPs available.
                    "exclude_file_operation_mcps": False,
                }
                if backend.get("base_url"):
                    backend_cfg["base_url"] = backend["base_url"]
                simplified.append(
                    {
                        "id": config.get("id", f"criteria_agent_{i}"),
                        "backend": backend_cfg,
                    },
                )

            coordination = {
                "enable_subagents": False,
                "broadcast": False,
                "checklist_criteria_preset": "evaluation",
            }
            if voting_sensitivity:
                coordination["voting_sensitivity"] = voting_sensitivity
            if voting_threshold is not None:
                coordination["voting_threshold"] = voting_threshold

            subagent_config = SubagentOrchestratorConfig(
                enabled=True,
                agents=simplified,
                coordination=coordination,
            )
            from massgen.precollab_utils import build_subagent_parent_context_paths

            parent_context_paths = build_subagent_parent_context_paths(
                parent_workspace=parent_workspace,
                agent_configs=agent_configs,
            )

            manager = SubagentManager(
                parent_workspace=criteria_workspace,
                parent_agent_id="criteria_generator",
                orchestrator_id=orchestrator_id,
                parent_agent_configs=simplified,
                max_concurrent=1,
                default_timeout=300,
                subagent_orchestrator_config=subagent_config,
                log_directory=log_directory,
                parent_context_paths=parent_context_paths,
            )

            prompt = self._build_generation_prompt(
                task,
                has_changedoc,
                min_criteria,
                max_criteria,
                has_planning_spec_context=has_planning_spec_context,
            )

            def _status_callback(subagent_id: str) -> Any | None:
                try:
                    return manager.get_subagent_display_data(subagent_id)
                except Exception:
                    return None

            if on_subagent_started:
                try:
                    subagent_log_path = None
                    if log_directory:
                        subagent_log_path = str(
                            Path(log_directory) / "subagents" / "criteria_generation",
                        )
                    on_subagent_started(
                        "criteria_generation",
                        prompt,
                        300,
                        _status_callback,
                        subagent_log_path,
                    )
                except Exception:
                    pass

            result = await manager.spawn_subagent(
                task=prompt,
                subagent_id="criteria_generation",
                timeout_seconds=300,
            )

            # Try to find criteria.json in output
            if log_directory:
                criteria = self._find_criteria_json(
                    log_directory,
                    min_criteria,
                    max_criteria,
                )
                if criteria:
                    self.last_generation_source = "subagent"
                    logger.info(
                        f"Loaded {len(criteria)} criteria from criteria.json",
                    )
                    return criteria

            # Try parsing from answer text
            if result.answer:
                criteria, aspiration = _parse_criteria_response(
                    result.answer,
                    min_criteria,
                    max_criteria,
                )
                if criteria:
                    self.last_generation_source = "subagent"
                    self.last_aspiration = aspiration
                    logger.info(
                        f"Parsed {len(criteria)} criteria from answer (aspiration: {aspiration})",
                    )
                    return criteria

            logger.warning("No valid criteria output found, using defaults")
            self.last_generation_source = "fallback"
            return get_default_criteria(has_changedoc=has_changedoc)

        except Exception as e:
            logger.error(f"Failed to generate criteria via subagent: {e}")
            self.last_generation_source = "fallback"
            return get_default_criteria(has_changedoc=has_changedoc)

    def _find_criteria_json(
        self,
        log_directory: str,
        min_criteria: int,
        max_criteria: int,
    ) -> list[GeneratedCriterion] | None:
        """Search for criteria.json in subagent logs."""
        from massgen.precollab_utils import find_precollab_artifact

        criteria_file = find_precollab_artifact(
            log_directory,
            "criteria_generation",
            "criteria.json",
        )
        if criteria_file is None:
            return None

        try:
            content = criteria_file.read_text()
            criteria, aspiration = _parse_criteria_response(
                content,
                min_criteria,
                max_criteria,
            )
            if criteria:
                self.last_aspiration = aspiration
                return criteria
        except Exception as e:
            logger.debug(f"Failed to parse {criteria_file}: {e}")

        return None
