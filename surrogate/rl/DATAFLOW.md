# DRL placement — data-flow (current pipeline)

**Mental model:** one GNN policy places ≤4 LiDARs on an *unseen* robot in a few fast
rollouts. It learns by imitating a greedy expert, then refines with RL. Everything is
scored against the **real raycaster**. Renders in VS Code (Markdown Preview Mermaid).

## 1. Whole pipeline (offline → train → deploy → score)

```mermaid
flowchart LR
  subgraph OFF["Offline (built once)"]
    G["robot graphs + meshes (shapes.py)"]
    S["surrogate.pt — GNN, IoU 0.54"]
    T["true footprint tables (cached raycast)"]
    O["OPT layouts — multi-start+LS (the ceiling)"]
  end
  OFF --> TR["TRAIN per fold:<br/>BC(greedy-on-true) then PPO(true reward)"]
  TR --> POL["trained policy<br/>(held-out robots = zero-shot)"]
  POL --> INF["INFER: best-of-N + verify-and-fallback"]
  INF --> EVAL["EVAL vs OPT (true-verified, multi-seed)"]
```

## 2. Training — how the policy learns

```mermaid
flowchart TD
  subgraph BC["1) Behaviour cloning (warm start)"]
    TT["true footprint table"] --> GT["greedy-on-TRUE teacher"]
    GT --> D["demos: state to action"]
    D --> CLONE["clone via cross-entropy"]
  end
  CLONE -->|"init policy"| RO
  subgraph PPO["2) PPO (refine)"]
    RO["rollouts on TRAIN robots"] --> RW["TRUE terminal reward (raycast):<br/>0.7*coverage - 0.3*cost, unclamped"]
    RW --> UP["update policy"]
    UP --> RO
    UP --> H["greedy rollout on HELD-OUT robots<br/>(zero-shot, true-verified)"]
  end
```

The **reward is the real raycaster**, not the surrogate. PPO-from-scratch fails, so BC
warms it up; teacher and reward both use TRUE coverage (they must agree).

## 3. One episode + inference (best-of-N + fallback)

```mermaid
flowchart TD
  R["reset(robot): GNN encode ONCE"] --> LOOP["place a sensor (<=4):<br/>node -> type -> orient, or STOP"]
  LOOP --> LOOP
  LOOP --> LAY["a layout"]
  LAY --> BON["best-of-N: greedy + 16 sampled rollouts"]
  BON --> VER["true-verify each (raycast), keep best"]
  VER --> CHK{"best &lt; 0.05 ? (collapse)"}
  CHK -->|"no"| OUT["final layout"]
  CHK -->|"yes"| FB["fallback: greedy-on-surrogate,<br/>true-verify, keep better"]
  FB --> OUT
```

best-of-N fixes the policy's peakiness (good layouts are in its distribution; greedy
decoding misses them); the fallback is a cheap safety net for rare collapses.

## 4. What the policy SEES (coverage-aware observation)

```mermaid
flowchart LR
  EMB["frozen GNN node embeddings<br/>(pool of 100 candidate nodes)"] --> POL
  USED["used-node mask + n_placed"] --> POL
  COV["running coverage_frac"] --> POL
  GAIN["per-node MARGINAL GAIN<br/>(surrogate decoder vs current coverage)"] --> POL
  POL["policy heads:<br/>node/STOP, type, orient, value"]
```

`coverage_frac` + `marginal_gain` are **state features** (what greedy sees) — they are
*inputs*, NOT the reward. This gives marginal-reasoning info without making a noisy
signal the reward.

## 5. Evaluation — the honest ceiling

```mermaid
flowchart LR
  P["policy (best-of-N)"] --> CMP
  GS["greedy-on-surrogate"] --> CMP
  GT["greedy-on-true"] --> CMP
  OP["OPT = multi-start + local-search<br/>(true optimum, NOT greedy)"] --> CMP
  CMP["true-verify all -> report % of OPT,<br/>mean +/- CI over seeds x robots"]
```

Numbers: fleet **OPT ≈ 0.316**; policy best-of-N ≈ **0.240 (~76% of OPT)**. Greedy is
**not** the ceiling — OPT beats greedy by 9–24%.

---

## The surrogate's THREE roles (and one non-role)

| role | where | uses |
|---|---|---|
| **Perception** | policy input | frozen GNN node embeddings (encoder) |
| **Marginal-gain feature** | coverage-aware obs | decoder predicts footprints → per-node gain |
| **Fallback proposer** | inference | greedy-on-surrogate when the policy collapses |
| ~~Reward~~ | — | **NOT used** — reward is the true raycaster |
