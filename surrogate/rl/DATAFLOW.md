# DRL placement — data-flow diagrams

Mermaid diagrams for the cross-robot zero-shot LiDAR placement RL. Renders in GitHub
and in VSCode (Markdown Preview Mermaid Support).

## 1. System map — where everything lives

```mermaid
flowchart TD
    subgraph OFFLINE["Offline assets (on disk)"]
        G["data/shapes — mesh + mounting graph"]
        S["data/surrogate.pt — FootprintGNN"]
        T["data/true_tables — raycast footprint tables"]
    end

    G --> SH["shapes.py — robot library"]
    S --> MO["model.py — GNN encode / decode"]
    T --> BT["bc_true.py — true-greedy teacher"]

    SH --> RM["reward.py — RewardModel"]
    MO --> RM
    MO --> POL["policy.py — PlacementPolicy"]

    RM -->|"Obs: frozen node_emb"| ENV["env.py — PlacementEnv"]
    ENV -->|"Obs"| POL
    POL -->|"Action"| ENV

    BT --> DEMO["BC demos: state to action"]
    BC["bc.py — surrogate-greedy teacher"] --> DEMO
    DEMO --> BCP["bc_pretrain — copy expert"]
    BCP -->|"init_policy"| POL

    ENV -->|"rollouts + reward"| PPO["train_ppo.py — PPO"]
    PPO -->|"update"| POL

    POL --> EV["evaluate.py / kfold.py"]
    EV -->|"true-verified"| RAY["CoverageEvaluator (real raycaster)"]
```

## 2. One episode — the inner loop

```mermaid
flowchart TD
    R["env.reset(robot)"] --> ENC["RewardModel.set_robot:<br/>graph --encode_graph(GNN)--> node_emb (N x 128)<br/>(computed once per robot)"]
    ENC --> OBS["Obs = node_emb, used_mask, n_placed"]

    OBS --> ACT["policy.act:<br/>pointer over nodes (mask used)<br/>then type, then orient<br/>(STOP forbidden if n_placed = 0)"]
    ACT --> STEP["env.step(Action)"]

    STEP --> DEC["RewardModel.step:<br/>decode footprint, OR into union<br/>(surrogate — no raycast)"]
    DEC --> UPD["used_mask[node]=True; n_placed += 1"]

    UPD --> DONE{"done? STOP or 4 sensors"}
    DONE -->|"no: reward = 0"| OBS
    DONE -->|"yes"| TERM["reward = fitness(union, cost)<br/>surrogate, or TRUE raycast if true_reward"]
    TERM --> OUT["layout -> scored by real raycaster for reporting"]
```

## 3. Training recipe — why two stages

```mermaid
flowchart TD
    subgraph P1["Phase 1 - Behavior Cloning (copy the expert)"]
        EXP["greedy expert<br/>surrogate OR true footprints"] --> D["demos: state to action"]
        D --> BCP["bc_pretrain (minimize cross-entropy)"]
        BCP --> COMP["policy starts COMPETENT (not random)"]
    end

    COMP -->|"init_policy"| RO

    subgraph P2["Phase 2 - PPO (improve by trial and error)"]
        RO["rollouts on TRAIN robots"] --> RW["rewards"]
        RW --> UP["ppo_update (nudge policy)"]
        UP --> RO
        UP --> HO["greedy rollout on HELD-OUT robots<br/>scored by REAL raycaster"]
    end

    NOTE["Winning combo: teacher = TRUE and reward = TRUE (must agree)"]
    HO --> NOTE
```

```mermaid
flowchart LR
    A["PPO from scratch"] --> B["random play -> reward ~ 0 everywhere"]
    B --> C["no gradient -> stuck at 0"]
    D["PPO from BC"] --> E["starts near good layouts"]
    E --> F["reward gradient exists -> improves"]
```

## 4. Shapes flowing through the networks

```mermaid
flowchart LR
    X["x: (N,6) pos+normal<br/>edges: (2,E)"] --> GNN["GNN encode_graph<br/>3x SAGEConv"]
    GNN --> EMB["node_emb (N,128)<br/>FROZEN, shared"]
    EMB --> RPATH["REWARD path - decode:<br/>node_emb[i] + sensor_emb + orient<br/>-> footprint (28800,)"]
    EMB --> PPATH["POLICY heads:<br/>node ptr (N), stop (1),<br/>type (3), orient (18), value (1)"]
```
