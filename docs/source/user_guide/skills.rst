==============================
Skills for AI Coding Agents
==============================

MassGen publishes **skills** that let AI coding agents (Claude Code, OpenAI Codex, GitHub Copilot, Cursor, and others) invoke MassGen directly. When your agent has the MassGen skill installed, it can spin up a multi-agent run, wait for results, and apply them. Learn more about the agent skills standard at `agentskills.io <https://agentskills.io/home>`_.

.. raw:: html

   <div style="text-align: center; margin: 20px 0;">
     <a href="https://github.com/massgen/skills" target="_blank" rel="noopener noreferrer" style="display: inline-block; padding: 12px 24px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; text-decoration: none; border-radius: 8px; font-weight: bold; font-size: 1.1em; box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4); transition: transform 0.2s, box-shadow 0.2s;">
       &#128736; Get the Skills on GitHub &rarr;
     </a>
   </div>

What Are Skills?
----------------

Skills are portable instruction bundles (a folder with a ``SKILL.md`` file) that teach AI agents how to perform specific workflows. The `SKILL.md format <https://agentskills.io/specification>`_ is an open standard supported by `40+ agent platforms <https://skills.sh>`_.

The **MassGen skill** gives your agent four modes:

.. list-table::
   :header-rows: 1
   :widths: 15 35 50

   * - Mode
     - Purpose
     - Output
   * - **General** (default)
     - Any task --- writing, code, research, design
     - Winner's deliverables + workspace files
   * - **Evaluate**
     - Critique existing work
     - ``critique_packet.md``, ``verdict.json``, ``next_tasks.json``
   * - **Plan**
     - Create a structured project plan
     - ``project_plan.json`` with task DAG
   * - **Spec**
     - Create a requirements specification
     - ``project_spec.json`` with EARS requirements

.. note::

   The skill will walk your agent through setup if needed, but things go smoother if you already have MassGen installed, an AI provider authenticated, and a config file ready. First-time setup requires human input (provider selection, API keys). See :doc:`/quickstart/installation` for setup instructions.

Installation
------------

Quick Install (All Agents)
^^^^^^^^^^^^^^^^^^^^^^^^^^

The fastest way to install across any supported agent:

.. code-block:: bash

   npx skills add massgen/skills

This works with Claude Code, Cursor, Codex, Windsurf, GitHub Copilot, Gemini CLI, Goose, Amp, and `40+ other agents <https://skills.sh>`_. See `Vercel's skills docs <https://vercel.com/docs/agent-resources/skills>`_ for details.

To install to a specific agent:

.. code-block:: bash

   npx skills add massgen/skills -a claude-code
   npx skills add massgen/skills -a codex
   npx skills add massgen/skills -a cursor

To install to all detected agents at once:

.. code-block:: bash

   npx skills add massgen/skills --all

Per-Agent Manual Installation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you prefer to clone and copy manually:

**Claude Code:**

.. code-block:: bash

   # Global (all projects)
   git clone https://github.com/massgen/skills.git /tmp/massgen-skills
   cp -r /tmp/massgen-skills/massgen ~/.claude/skills/massgen

   # Per-project (committed to your repo)
   mkdir -p .claude/skills
   cp -r /tmp/massgen-skills/massgen .claude/skills/massgen

Then invoke with ``/massgen`` in Claude Code.

**OpenAI Codex:**

.. code-block:: bash

   git clone https://github.com/massgen/skills.git /tmp/massgen-skills
   cp -r /tmp/massgen-skills/massgen ~/.codex/skills/massgen

Then invoke with ``$massgen`` in Codex.

**GitHub Copilot (VS Code):**

.. code-block:: bash

   git clone https://github.com/massgen/skills.git /tmp/massgen-skills
   cp -r /tmp/massgen-skills/massgen .github/skills/massgen

Then use ``/skills`` in Copilot chat.

**Other agents:**

Any agent supporting the ``SKILL.md`` standard can use MassGen skills. Copy the ``massgen/`` directory from `the repo <https://github.com/massgen/skills>`_ into your agent's skill discovery path (typically ``~/.agents/skills/``).

Prerequisites
^^^^^^^^^^^^^

1. MassGen installed (``pip install massgen``)
2. At least one AI provider authenticated (API key or login-based auth like ``claude login``)
3. A MassGen config file (``.massgen/config.yaml``) --- run ``massgen --quickstart`` to create one

How It Works
------------

When your agent invokes the MassGen skill, it follows this workflow:

1. **Scope** --- determine the mode (general, evaluate, plan, spec) and what the run covers
2. **Context** --- write a context file describing the task, constraints, and expectations
3. **Criteria** --- use defaults or write custom evaluation criteria
4. **Prompt** --- fill in the mode-specific prompt template
5. **Run** --- launch MassGen in ``--automation`` mode (background), optionally open the web viewer
6. **Parse** --- read the structured output from the winning agent
7. **Apply** --- ground the results in your task system and execute

The skill includes prompt templates, context file guides, and output parsing instructions for each mode. Your agent reads these reference files and follows them step by step.

Skill Contents
--------------

The skill repo at `github.com/massgen/skills <https://github.com/massgen/skills>`_ contains:

::

   massgen/
   +-- SKILL.md                              # Main skill instructions
   +-- references/
       +-- criteria_guide.md                  # How to write evaluation criteria
       +-- general/
       |   +-- workflow.md                    # General mode guide
       |   +-- prompt_template.md             # General prompt template
       +-- evaluate/
       |   +-- workflow.md                    # Evaluate mode guide
       |   +-- prompt_template.md             # Evaluation prompt template
       +-- plan/
       |   +-- workflow.md                    # Plan mode guide
       |   +-- prompt_template.md             # Planning prompt template
       +-- spec/
           +-- workflow.md                    # Spec mode guide
           +-- prompt_template.md             # Spec prompt template

Keeping Skills Updated
----------------------

The skills repo is automatically synced from the main MassGen repository on every merge to ``main``.

.. code-block:: bash

   # If installed via npx
   npx skills update

   # If installed via git clone
   cd /tmp/massgen-skills && git pull
   cp -r /tmp/massgen-skills/massgen ~/.claude/skills/massgen   # or your agent's path
