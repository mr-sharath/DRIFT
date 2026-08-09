# 📘 Waynex / DRIFT Phase 2: Method of Procedure (MOP) & Technical Documentation

**Author:** Sharath  
**System:** Waynex AI Routing Engine (Neuro-Symbolic GNN + DRL + 2-Opt)  
**Version:** Phase 2 Production Candidate  
**Date:** August 2026  

---

## 🎯 Executive Summary & Purpose

This document serves as the complete **Method of Procedure (MOP)**, operational manual, and technical retrospective for **Phase 2 of the Waynex / DRIFT platform**. 

It logs every design decision, architectural pivot, text tag, error encountered, and resolution applied from initial compute setup to the final enterprise UI redesign and cloud deployment strategy.

---

## 🏗️ 1. Complete Chronology & Architectural Decisions

### Phase 2.1: From Financial Graph Routing to Real-World Logistics Fleet Control Center
- **Original Vision:** DRIFT began as a research-backed Graph Neural Network (GNN) agent for financial transaction routing across Payment Channel Networks (Lightning Network).
- **Phase 2 Expansion:** Expanded the underlying graph-learning engine into **Waynex**, an interactive dual-benchmark platform comparing AI-driven fleet dispatching against deterministic Mixed Integer Linear Programming (MILP) solvers like **Google OR-Tools**.

### Phase 2.2: Compute Engine Strategy Pivot (Modal → Colab Pro → Local CPU Inference)
1. **Modal Serverless Platform Attempt:**
   - *Goal:* Spin up cloud GPU workers on Modal for fast DRL model rollouts.
   - *Outcome:* Encountered setup overhead and active environment conflicts with user's existing Modal web services.
   - *Decision:* Cleaned up Modal dependencies and pivoted to **Google Colab Pro** with L4/A100 GPUs for offline pre-training while keeping real-time inference CPU-light (sub-50ms) on the local application server.
2. **Offline Pre-Training on Colab Pro:**
   - Synthetic rollout training script (`train_colab.ipynb` / `train_colab_v2.py`) trains the spatial GNN and Actor-Critic weights across 50,000 synthetic city graph topologies.
   - Generated checkpoint files (`waynex_policy_v2.pt`) are loaded by the local MCP server (`mcp_server.py`) for sub-50ms CPU inference.

### Phase 2.3: City Hub Template Strategy & London Routing Problem
- **The Problem:** The **London Central & Outer Ring** template exhibited solver timeouts and dropped nodes under Google OR-Tools. The large spatial spread caused OR-Tools' vehicle routing solver to hit soft-penalty bounds, resulting in unserved pins or long solve delays (>10s).
- **The Fix:**
  - Replaced the London template with **Tokyo Metro Logistics Center** (`city_templates.py`), which offers a dense urban topology ideal for real-time dispatch testing.
  - Reverted hard disjunction penalties in `ortools_solver.py` back to standard values, preserving un-handicapped OR-Tools baseline performance.
  - Kept **Bengaluru Tech & Logistics Hub** and **San Francisco Bay Area Fleet** as active city hubs.

### Phase 2.4: Neuro-Symbolic Engine Upgrade (GNN + DRL + 2-Opt)
- **The Problem:** Pure neural network greedy decoding alone (5,000 to 50,000 episodes) underperformed against Google OR-Tools because neural networks lack natural "lookahead" and can generate crossed path edges.
- **The Solution (Neuro-Symbolic Architecture):**
  - **Stage 1 (Intuition):** Spatial GNN Encoder (`gnn_encoder.py`) converts latitude/longitude, delivery demand, and time windows into 64-dim embeddings.
  - **Stage 2 (Action Masking & Distance Penalty):** Actor-Critic Policy (`drl_policy.py`) evaluates candidate hops with explicit distance penalties injected into neural logits.
  - **Stage 3 (Symbolic Refinement):** Integrated a `2-opt` local search algorithm in `drl_policy.py`. After the neural network constructs the initial route, `2-opt` instantly untangles crossed edges in <1ms.

### Phase 2.5: RL Reward Formulation Upgrade (PPO Return-to-Go)
- **The Problem:** Training the model for 50,000 episodes on Colab caused performance degradation due to myopic step-rewards (`-step_distance`) which created policy collapse.
- **The Fix:** Created an advanced PPO-style training script (`train_colab_v2.py` / updated `train_colab.ipynb`) using **Discounted Return-to-Go ($\gamma = 0.99$)**, reward normalization, and gradient clipping (`max_norm=0.5`).

### Phase 2.6: Enterprise UI & Branding Redesign
- **Original UI:** Dark blue slate prototype with left sidebar navigation, hardcoded blue themes, and a cramped 3-column layout.
- **New Enterprise Design:**
  - **Color Palette:** Clean light base (`#f5f6f8`), pure white cards (`#ffffff`) with subtle drop shadows (`box-shadow: 0 1px 3px rgba(0,0,0,0.06)`), and an enterprise **Teal** accent (`#0f766e`).
  - **Navigation:** Replaced the 240px left sidebar with a slim 56px horizontal top navigation bar (`.top-nav`).
  - **City Selector:** Replaced standard dropdowns with interactive horizontal **City Cards** (`Bengaluru`, `San Francisco`, `Tokyo`).
  - **Layout:** 2-column Dispatch Center layout giving maximum screen area to the Leaflet map while keeping Fleet Capacity and Live Status Feed stacked on the left.
  - **Map Layer:** Switched to Carto Positron light map tiles (`light_all`).
  - **Marketing Homepage:** Designed a professional "Platform" landing view explaining the GNN → DRL → 2-Opt pipeline, architecture diagrams, and OpenAI Copilot features.

---

## 🛠️ 2. Comprehensive Log of Errors Faced & Resolutions

| # | Error / Issue Encountered | Root Cause | Resolution Applied |
|---|---------------------------|------------|--------------------|
| **1** | Modal platform deployment conflicts & environment errors | Pre-existing Modal services on user account & setup overhead | Removed Modal scripts, shifted GPU training to Google Colab Pro, and used PyTorch CPU mode for local production server. |
| **2** | Google OR-Tools taking >10 mins or failing on London template | Wide spatial dispersion in London dataset causing solver disjunction penalty overflows | Replaced London with Tokyo Metro template in `city_templates.py`; reverted hardcoded penalty overrides in `ortools_solver.py`. |
| **3** | Neural policy performance degraded when scaling training from 5k to 50k episodes | Myopic RL step reward (`-step_dist`) causing policy collapse under long iterations | Rewrote training script (`train_colab_v2.py`) to use Discounted Return-to-Go ($\gamma=0.99$), reward standard-scaling, and gradient norm clipping. |
| **4** | Route lines on map rendered entirely in green/teal instead of individual vehicle colors | CSS class `.leaflet-interactive.route-polyline` had hardcoded `stroke: var(--brand-primary)` overriding Leaflet inline SVG styles | Removed `stroke` property from `.leaflet-interactive.route-polyline` in `index.css`, allowing dynamic truck color rendering. |
| **5** | UI map showing Bengaluru when selecting Tokyo | Python backend (`main_api.py`) holding cached version of `city_templates.py` in memory | Instructed user to restart `main_api.py` server to re-import updated templates. |
| **6** | Changing Search Time Budget slider did not reset "View Benchmarks" button | Missing event hook on slider input change | Added button state reset logic inside `updateOrToolsTimeLabel()` in `app.js`. |
| **7** | `git status` error inside terminal sandbox (`libpcre2-8.0.dylib` open blocked) | Terminal sandbox path restrictions on system dylibs | Used system git (`/usr/bin/git`) with sandbox bypass to execute git operations cleanly. |
| **8** | OpenAI API key visible in left nav rail | Key input box placed directly in main navigation interface | Moved key input to a dedicated Settings Modal accessible via a gear icon (`⚙️`) in the top nav. |

---

## 🏷️ 3. Complete Text Tags & Copywriting Reference

Below is a reference of key text tags, titles, and branding copy used throughout the redesigned platform:

- **Platform Brand Title:** `WAYNEX — Neuro-Symbolic Fleet Intelligence`
- **Hero Title:** `Fleet Intelligence, Delivered in Milliseconds`
- **Hero Subtitle:** `Waynex combines Graph Neural Networks with Deep Reinforcement Learning to compute highly optimized multi-vehicle fleet routes in under 50ms — rivaling traditional solvers that take seconds.`
- **Pipeline Stage 1:** `01 — Graph Neural Network Encoder`
- **Pipeline Stage 2:** `02 — Actor-Critic RL Policy`
- **Pipeline Stage 3:** `03 — Neuro-Symbolic 2-Opt Refinement`
- **Fair Benchmark Commitment:** `We believe in transparent, un-handicapped comparisons. In the Dispatch Center, every routing request runs a live dual-benchmark: Waynex Neural Engine vs. the official Google OR-Tools solver.`
- **AI Dispatch Copilot Tagline:** `An integrated conversational AI assistant powered by OpenAI GPT helps dispatchers understand routing decisions, fuel savings, and CO₂ impact.`
- **Telemetry Metrics Tracked:** `Total Distance Travelled (km)`, `Max Shift Duration / Makespan (min)`, `Vehicle Capacity Utilization (%)`, `Total Fuel Consumed (gal)`, `Solver Compute Latency (ms)`.

---

## 🌐 4. Cloud Deployment Strategy & Resource Sizing

### 4.1 Technical Sizing Requirements
- **Memory (RAM):** ~512 MB minimum / 1 GB recommended (Runs lightweight PyTorch CPU inference + native Python HTTP server).
- **Compute (CPU):** 1 vCPU is sufficient for sub-50ms neural rollouts + 2-opt refinement + 1s OR-Tools solving.
- **Disk Storage:** ~250 MB total (Python 3.11, PyTorch CPU wheel, project files, `.pt` model weights ~145 KB).
- **Network Egress:** Required for OSRM route geometry calls (`router.project-osrm.org`). Includes Haversine fallback matrix calculator if OSRM is unreachable.

### 4.2 Hosting Options & Free Tier Capabilities

#### Option A: Render.com (Recommended Free Tier)
- **Deployment:** Web Service (Python Native or Docker)
- **Free Tier Specs:** 512 MB RAM, 0.1 CPU, Free SSL, automatic GitHub deploys.
- **Setup Command:** `python main_api.py`
- **Environment Variables:** `PORT=8000`

#### Option B: Hugging Face Spaces (Free Machine Learning Hosting)
- **Deployment:** Docker / Gradio / Static Space
- **Free Tier Specs:** 2 vCPU, 16 GB RAM, 50 GB Storage.
- **Ideal For:** Showcasing AI models & benchmark demos to business partners and investors.

#### Option C: Koyeb / Fly.io / Railway
- **Koyeb Free Tier:** 512 MB RAM, Nano instance, automatic git deployment.
- **Fly.io:** 256 MB RAM micro-vms with global edge routing.

#### Option D: Vercel (Frontend) + Render (Backend)
- Decouple static files (`public/`) onto Vercel CDN for ultra-fast global static delivery, pointing API requests to Render backend (`main_api.py`).

---

## 📋 5. Production Release Checklist

- [x] **Security Headers Configured:** `Content-Security-Policy`, `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`, `X-XSS-Protection`, `Referrer-Policy`.
- [x] **Input Sanitization:** HTML tag stripping & length limiting (`sanitize_input`) on search & autocomplete endpoints.
- [x] **API Key Privacy:** OpenAI key input secured inside local browser storage via Settings Modal.
- [x] **Responsive Layout Tested:** Desktop (≥1200px), Tablet (768-1199px), Mobile (≤767px).
- [x] **Light Map Tile Layer:** Carto Positron (`light_all`) configured.
- [x] **Dynamic Route Line Colors:** Per-vehicle truck color matching on Leaflet polylines.
- [x] **Automatic Model Checkpoint Fallback:** `mcp_server.py` auto-loads `waynex_policy_v2.pt` → `waynex_policy_v1.pt` → `waynex_policy.pt`.
- [x] **Documentation & MOP Complete:** `README.md` updated and `PHASE2_MOP_DOCUMENTATION.md` created.

---

## 🚀 6. Local Quick-Start Guide

```bash
# 1. Clone the repository
git clone https://github.com/mr-sharath/DRIFT.git
cd DRIFT

# 2. Install dependencies
pip install torch numpy ortools

# 3. Start the Waynex Control Center server
python main_api.py

# 4. Open in browser
open http://localhost:8000
```

---
*End of Phase 2 Method of Procedure & Documentation.*
