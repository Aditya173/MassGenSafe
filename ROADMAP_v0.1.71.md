# MassGen v0.1.70 Roadmap

**Target Release:** March 30, 2026

## Overview

Version 0.1.70 focuses on running MassGen as a cloud job on Modal.

---

## Feature: Cloud Modal MVP

**Issue:** [#982](https://github.com/massgen/MassGen/issues/982)
**Owner:** @ncrispino

### Goals

- **Cloud Execution**: Run MassGen jobs in the cloud via `--cloud` option on Modal
- Progress streams to terminal, results saved locally under `.massgen/cloud_jobs/`

### Success Criteria

- [ ] Cloud job execution functional on Modal
- [ ] Progress streaming and artifact extraction working

---

## Related Tracks

- **v0.1.69**: WebUI Automation & Skill Mode — auto-start, CLI flags with `--web`, MassGen skill in WebUI, Gemini CLI provider ([#1032](https://github.com/massgen/MassGen/pull/1032))
- **v0.1.71**: OpenAI Audio API ([#960](https://github.com/massgen/MassGen/issues/960))
