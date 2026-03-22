# MassGen v0.1.67 Roadmap

**Target Release:** March 23, 2026

## Overview

Version 0.1.67 focuses on running MassGen as a cloud job on Modal.

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

- **v0.1.66**: Step Mode — `--step` CLI for external orchestrators, Codex Windows fixes ([#1011](https://github.com/massgen/MassGen/pull/1011), [#1010](https://github.com/massgen/MassGen/pull/1010))
- **v0.1.68**: OpenAI Audio API ([#960](https://github.com/massgen/MassGen/issues/960))
