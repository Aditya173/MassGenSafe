# MassGen v0.1.75 Roadmap

**Target Release:** April 11, 2026

## Overview

Version 0.1.75 focuses on running MassGen as a cloud job on Modal.

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

- **v0.1.74**: Checkpoint Improvements & Tool Call Fixes — checkpoint MCP enhancements, duplicate tool call fix ([#1050](https://github.com/massgen/MassGen/pull/1050))
- **v0.1.76**: OpenAI Audio API ([#960](https://github.com/massgen/MassGen/issues/960))
