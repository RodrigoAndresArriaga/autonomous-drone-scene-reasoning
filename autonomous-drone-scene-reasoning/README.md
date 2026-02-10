# Autonomous Drone Scene Reasoning

This project implements a drone-based reasoning agent that evaluates shared physical safety step-by-step, using non-planning, shared-safety reasoning with optional exit context to determine whether a human can safely follow a drone.

---

## 1. What problem is solved

Autonomous drones operating in real-world environments (industrial inspection, disaster response, environmental monitoring) often fail before control — at the scene understanding and decision stage, especially when humans are present.

Current systems:

- Detect objects but do not reason about shared physical risk
- Evaluate safety only for the robot, not for humans following its path
- React to hazards but cannot explain why a route is unsafe
- Lack human-aware and physics-aware guidance logic

This leads to unsafe navigation decisions, misleading guidance in human-assisted scenarios, poor trust in autonomy, and high false positives / false negatives in hazard assessment.

**Objective:** Implement a vision-based reasoning agent that enables a drone to:

1. Understand its environment from egocentric video
2. Reason about physical hazards: terrain, obstacles, environmental hazards, and human presence as a traversal constraint
3. Evaluate shared-path safety: is the locally observed path safe for the drone, and is it safe for a human to follow (particularly when guidance toward a safer zone is implied)?
4. Recommend a navigation or guidance action that accounts for both agents
5. Explain the reasoning behind that decision

The system focuses on **reasoning and decision support**, not flight control.

The core contribution of this project is the explicit separation between drone-safe traversal and human-safe traversal, enabling the agent to prevent humans from following robots into paths that are physically plausible for the robot but unsafe for a person.

---

## 2. What the agent guarantees

For each input video segment or frame window, the system produces:

1. **Hazard assessment** — Obstacle proximity, terrain stability / passability, environmental hazards (e.g., debris zones, plumes), and human presence as a physical constraint.

2. **Dual safety evaluation**
   - **Drone path safety:** Safe / Caution / Unsafe
   - **Human-follow path safety:** Safe to follow / Follow with caution / Unsafe for human traversal  
   This explicitly distinguishes drone-only affordances (e.g., flying over gaps, tight clearances) from human affordances (walking, balance, body clearance).

3. **Optional exit context (abstract, non-planning)** — Directional context when a safer zone or exit is implied. Not required for path safety classification. Used only to interpret guidance decisions, not to generate them. Does not perform exit discovery, mapping, or route selection. Path safety classification is always computed independently of exit context.

4. **Risk map (abstract)** — Directional risk scores (forward, left, right, up) that incorporate human vs drone constraints.

5. **Navigation & guidance recommendation** — Proceed and guide human / Proceed but do not guide human / Reroute before guiding / Hold position. Guidance decisions are made step-by-step. Shared-path safety classification always precedes any consideration of exit or goal direction.

6. **Textual reasoning explanation** — Clear, physically grounded explanations (e.g., “While the drone can safely pass over the debris field, the uneven terrain presents a tripping hazard for a human. The recommended action is to reroute before guiding a person through this area.”).

---

## 3. What it does NOT do (non-goals)

- No low-level control (PID, MPC, flight stack)
- No real-time autonomy requirement
- No global mapping, SLAM, or path planning
- No medical, emotional, or distress inference
- No generative media or simulation realism focus

Humans are treated strictly as physical agents with traversal constraints.

These exclusions are intentional to keep the system focused on interpretable, judge-verifiable physical reasoning rather than control performance or simulation fidelity.

---

## 4. High-level pipeline

```mermaid
flowchart TD
    A[Egocentric Drone Video] --> B[Video Analytics Reasoning Agent]
    B --> C[Egocentric Scene Understanding with Cosmos Reason 2]
    C --> D[Hazard and Constraint Identification]
    D --> E[Physical Plausibility and Safety Filtering]
    E --> F[Dual Risk Map Drone vs Human]
    F --> G[Optional Exit Context Update non-planning]
    G --> H[Navigation and Guidance Recommendation]
    H --> I[Textual Reasoning Explanation]
```

**In short:** Vision → Reasoning → Shared safety decision → Explanation.

---

## 5. Inputs / Outputs

**Inputs**

- Egocentric drone video (simulated or recorded)
- Optional metadata (altitude, camera angle)

**Outputs**

- Hazard assessment (obstacles, terrain, environmental hazards, human as constraint)
- Dual safety evaluation (drone path + human-follow path)
- Optional exit context (directional, non-planning)
- Risk map (directional scores for drone vs human)
- Navigation & guidance recommendation
- Textual reasoning explanation

---

## 6. How Cosmos Reason 2 is used

**Egocentric Social & Physical Reasoning (core reasoning engine)**  
This project uses the Egocentric Social & Physical Reasoning recipe from the Cosmos Cookbook to perform embodied, robot-centric reasoning over egocentric video. The agent queries Cosmos Reason 2 to extract hazards, human presence, and spatial context for physical safety evaluation.

**Video Search & Summarization (continuous video pattern)**  
This project adapts the Video Search & Summarization recipe pattern for continuous video analytics. Rather than summarizing content, it queries Cosmos Reason 2 repeatedly to assess dynamic hazards and update shared safety decisions.

**Physical Plausibility Prediction (inspiration only)**  
The project’s physical plausibility logic is inspired by the Physical Plausibility Prediction recipe, which demonstrates how to judge consistency with physical laws. The project adopts this principle in its reasoning chain by applying constraint logic rather than numeric physics scoring.

**Intelligent Transportation post-training (not used)**  
This project does not use the Intelligent Transportation post-training recipe. Post-training biases Reason 2 to a specific labeled domain (e.g., traffic), which conflicts with its requirement for open-ended physical hazard reasoning and interpretability. This project intentionally uses Cosmos Reason 2 in inference mode; post-training on a fixed hazard taxonomy would narrow the model’s effective hazard vocabulary and reduce generalization to unseen or compound physical risks.

---

## 7. How to run a demo

**Prerequisites**

- Cosmos Cookbook cloned (for docs and recipe patterns); Cosmos Reason 2 installed (e.g., clone the repo and run the provided sample inference script).
- GPU: 2B model needs ~24GB VRAM; 8B needs ~32GB.

**Demo format**

- Short clips (5–10 seconds), egocentric POV, no HUD.
- Clips are run through the agent and end at classification.
- The intended demo set follows the blueprint scenarios: **S1** (Class A — safe for drone and human), **S2** (Class B — gap / broken floor), **S3** (Class B — narrow passage), **S4** (Class C — dynamic / unstable area).

**Running a demo**

- Run the project’s smoke test or single-clip pipeline: feed one short indoor clip through Cosmos Reason 2 and obtain stable textual output for hazards, drone safety, human-follow safety, recommendation, and explanation.
- Recommended setup: develop and run on a GPU machine (e.g., remote SSH). Optionally, a remote Reason 2 inference API (e.g., vLLM server) may be used once the core agent loop is stable. The reference implementation prioritizes clarity and reproducibility over deployment complexity.

When you run a demo, you should see: detected hazards, drone path safety, human-follow safety, recommendation, and a physically grounded explanation.
