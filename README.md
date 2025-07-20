# 🧠 **DRIFT System Architecture**

> *“An Intelligent Transaction Routing Framework using GNNs and Deep Reinforcement Learning”*

---

## 🔁 0. Motivation

Traditional transaction routing (e.g., in banking, telecom, blockchain) is:

* **Rule-based** (e.g., round robin, FIFO)
* **Hard-coded thresholds**
* **Fails under dynamic or high-load conditions**

What if we had a **learned agent** that:

* Understands the **graph structure** (banks/nodes, transaction limits),
* Adapts routing based on **latency, cost, risk**,
* Learns through simulation using **deep RL** and **graph encoding**?

That’s what we’ll build.

---

## 🧱 1. System Overview

```
graph TD
    T[Transaction Generator] --> E[Graph Environment]
    E -->|State s_t| A[Graph Encoder (GNN)]
    A --> P[RL Policy (DRL Agent)]
    P -->|Action a_t| E
    E -->|Reward + New State| P
    P -->|Learned weights| I[Inference Engine]
    I -->|API Request| B[Fastify Backend]
    B -->|WebSocket| F[Interactive Frontend]
```

---

## 🧩 2. Components (Deep Dive)

### 🟢 A. **Graph Simulator**

* Simulates a network of banks/agents as a **dynamic graph**: nodes = banks, edges = payment corridors.
* Nodes have features: capacity, fee, location, etc.
* Edges have constraints: fee, delay, bandwidth.
* **`networkx` + synthetic data generator**

---

### 🔵 B. **GNN Encoder**

* Learns representations of each node (bank) and edge (transaction rule).
* Input: graph `G(V,E)` with features
* Output: embeddings for nodes/edges
* Architecture:

  * `Graphormer`, `GIN`, or `GraphSAGE`
  * Implemented in `PyTorch Geometric`

---

### 🟠 C. **DRL Agent**

* Uses policy gradient (PPO or A2C) to select the next routing decision
* Observes:

  * Current graph state `s_t`
  * Transaction queue
* Chooses:

  * Next hop or action `a_t`
* Learns:

  * Policy `π(a|s)` using `CleanRL`
* Reward:

  * Delivery success, cost, speed

---

### 🟡 D. **Training Loop**

* Plug GNN encoder into RL policy network
* Define custom environment using `Gymnasium`
* Train over thousands of simulated episodes
* Use `WandB` or `TensorBoard` for metrics

---

### 🟣 E. **Inference Module**

* Export trained policy to **ONNX**
* Run **inference locally** via `onnxruntime`
* Accept real transaction requests → recommend routing path

---

### 🟤 F. **Demo Interface**

* Simple **interactive graph UI** (drag/drop nodes, simulate congestion)
* Real-time routing visualized
* Built with:

  * `SvelteKit` + `D3.js` (or simple HTML + JS)
  * `Fastify` backend with socket or REST
  * `Firebase` or `GCP Cloud Run` hosting

---

## 🧪 3. Research Evaluation Plan

| Metric              | Why It Matters               |
| ------------------- | ---------------------------- |
| 🕒 Avg latency      | Measures routing efficiency  |
| 💰 Transaction cost | Penalizes expensive hops     |
| ✅ Completion %      | Measures success rate        |
| 🔄 Generalization   | Evaluate on unseen networks  |
| 🧠 Ablation Study   | Prove GNN improves RL policy |

---

## 🧠 4. What's Novel Here?

* **Combining GNN + RL for end-to-end adaptive routing**
* **Real-time inference over constrained dynamic networks**
* **Fully open-source + deployable** (not just academic)
* **Optimized with user-tuned reward shaping**
* **Fits smart finance, crypto, and edge network use cases**

---
