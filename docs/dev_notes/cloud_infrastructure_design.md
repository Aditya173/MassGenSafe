# Cloud Infrastructure Decision Document

**Author:** MassGen Team
**Date:** 2026-02-18
**Status:** Design Phase

## Overview
This decision doc evaluates cloud platforms for running MassGen agents and provides a recommendation for the infrastructure.

## Platform Comparison Matrix

| | Modal | E2B | Cloud Run | Kubernetes |
|---|---|---|---|---|
| Cold Start Latency | sub-second | sub-second | ~2s | variable |
| Cost Model | Per-sec | Base Fee + Usage Fee | Usage Fee | Per Cluster Per Hour |
| Isolation | Container + gVisor Sandbox | FirecrackerVM | Container | Container |
| Networking | Tunnels (live TCP ports) + Clusters (IPv6 Private Network) + HTTP endpoints| HTTP inside sandbox | HTTPS endpoints | Custom Container Network Interface (including cluster networking) |
| State Persistence | Modal Volumes and Snapshots | Filesystem + Snapshots + Sandbox Pause/Resume | Cloud Storage Fuse / no snapshots | Persistent Volume Claims (PVC) |
| DX | Python SDK Centric | Python SDK | Moderate Setup | High Ops Burden |
| MassGen Container Compatibility | No native Docker-in-Docker support but has native Sandbox for agent tool execution containers (requires significant code refactoring for MVP) | Yes | No native Docker-in-Docker support | No native Docker-in-Docker support |

## Recommended Platform
Start with **Modal**
Rationale:
1. Serverless: MassGen does not need to be running 24/7, only on-demand.
2. Python SDK
3. Lowest User Friction: Little to no devops and no cloud/container knowledge required.
4. Cost: free resources up to $30/month, low CPU and Memory costs.

Note: For MVP, E2B would require the least amount of development work with little to no refactoring, thus easiest "hello world" cloud agent. However, E2B is the least optimal for the long run due to its cost model, which I think would drive a lot of users away, especially if it is our only cloud offering. The free version of E2B only allows for up to 1hr sandbox sessions and up to 20 concurrently running sandboxes, making it unsuitable for MassGen uses especially in the long run with Topology B. E2B pro version has a base fee of $150/month + usage cost.

## Container Topology Recommendation
Topology A: One Big Container

```
┌────────────────────────────────────┐
│ Orchestrator Container             │
│                                    │
│  Agent 1 (subprocess)              │
│  Agent 2 (subprocess)              │
│  Agent 3 (subprocess)              │
└────────────────────────────────────┘
```

Topology B: Container Per Agent

```
┌────────────────────────────────────┐
│ Orchestrator Container             │
└──────────┬─────────────────────────┘
           │
    ┌──────┴──────┬──────────┐
    ▼             ▼          ▼
┌─────────┐  ┌─────────┐  ┌─────────┐
│Agent 1  │  │Agent 2  │  │Agent 3  │
│         │  │         │  │         │
│Sandbox  │  │Sandbox  │  │Sandbox  │
└─────────┘  └─────────┘  └─────────┘
```

Start with Topology A to test cloud setup for MVP. After successful MVP, migrate to Topology B.

Rationale:
1. Fastest path to cloud MVP.
2. This process allows for supporting both options, giving users the flexibility to choose which one suits their needs.
3. Topology B is significantly more complex. Requires design for networking/communication and storage.

## Known Risks
The main risk is lots of refactoring to avoid docker-in-docker as serverless computing options do not natively support docker-in-docker, which would cause vendor lock-in.

## Comparison to Other Similar Agent Harnesses

| Platform | Architecture |
|---|---|
| OpenAI Codex | Cloud-only, ephemeral container per task. |
| OpenHands | Three methods of running: (1) locally in LocalWorkspace, (2) sandboxed locally using Docker container (DockerWorkspace), (3) sandboxed remotely using remote container via HTTP (RemoteAPIWorkspace). Agents execute tools in the same container, not a separate container. `AgentController`s are always run locally, while agent servers live in the (local/remote) workspaces |

The cloud infrastructure goals for MassGen differs from existing frameworks in two key ways:
1. Existing frameworks are essentially single agents. MassGen wants communication between orchestrator and worker agents.
2. MassGen Cloud wants the orchestrator to be containerized, running in the cloud and have the ability to spawn containerized agents (under Topology B).

