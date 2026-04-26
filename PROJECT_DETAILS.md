# 🎭 OrchestraAI: Advanced Multi-Agent Orchestration Framework

**OrchestraAI** is a comprehensive Reinforcement Learning (RL) research platform and production framework designed to solve the **"Orchestration Gap"** in LLM systems. It focuses on training a high-level **Manager Agent** to autonomously supervise, evaluate, and refine the outputs of a heterogeneous pool of **Worker Agents**.

---

## 🚀 1. The Core Philosophy: "Quality through Verification"

In traditional multi-agent systems, agents are often assumed to be reliable. OrchestraAI assumes **systemic unreliability**. 
- **Workers** are treated as "Black Boxes" with stochastic performance.
- **Managers** must treat tokens as a finite currency, balancing the cost of checking work against the risk of passing failures.
- **The Objective**: Maximize the "Total Quality" of the project while minimizing "Operational Cost" (Tokens).

---

## 🏗️ 2. System Architecture

The framework is architected into three distinct layers:

### A. The Environment Layer (`ManagerWorkerEnv`)
A complex Gymnasium-compatible environment that simulates a high-stakes software project or research task.
- **Dynamic Task Graph**: Tasks are broken into sequential or parallel subtasks.
- **Budgeting System**: Every action (including doing nothing) has a cost. The Manager must survive within the `token_budget`.
- **Observation Space**: A multi-discrete dictionary containing:
    - `worker_states`: Health, current subtask, and skill levels.
    - `task_state`: Progress per subtask and overall project completion.
    - `budget_info`: Normalized remaining tokens and time steps.
    - `last_action_results`: Feedback from the previous step.

### B. The Policy Layer (Manager Agent)
OrchestraAI supports multiple "Brains" for the Manager:
1. **PPO (Proximal Policy Optimization)**: Trained using Stable-Baselines3. This agent learns optimal heuristics, such as "Don't check a Senior worker's output if the budget is low."
2. **Parallel Strategy**: A deterministic agent designed for high-throughput, fanning out as much work as possible.
3. **Heuristic Strategy**: A safety-first agent that checks every single output before approval.

### C. The Worker Layer (Execution Units)
Workers are standalone LLMs integrated via the Hugging Face ecosystem.
- **Diversity**: Mix of `SmolLM` (for fast, cheap tasks) and `Llama-3` (for complex logic).
- **SFT (Supervised Fine-Tuning)**: Workers in Phase 2 are fine-tuned using the **TRL** library on a custom dataset of 1,000+ "Success vs. Failure" examples to ensure they simulate hallucinations realistically.

---

## 🔍 3. The Hallucination & Quality Engine

The project includes a sophisticated simulator for LLM failure modes:
- **Quality Score [0.0 - 1.0]**: Every subtask output has a hidden quality score determined by worker skill + luck.
- **Failure Modes**:
    - `hallucination`: The worker provides confident but false information.
    - `off_task`: The worker ignores instructions and talks about something else.
    - `incomplete`: The worker stops halfway.
- **The "Check" Action**: When a Manager "checks" a worker, the environment reveals the hidden `quality_score`. Without a check, the Manager is "flying blind."

---

## 📊 4. Reinforcement Learning Details

### Reward Shaping
The Manager's reward at each step is calculated as:
$$R = (Q_{approved} \times 10) - (T_{spent} \times 0.01) + B_{completion}$$
Where:
- $Q_{approved}$: Quality of the approved subtask.
- $T_{spent}$: Tokens consumed in that step.
- $B_{completion}$: A massive bonus for completing the entire task graph.

### Training Phases
- **Phase 1 (Foundations)**: Training the Manager on a static Task Library.
- **Phase 2 (Co-Evolution)**: Training the Manager alongside fine-tuned Workers who exhibit specific failure patterns.

---

## 🛠️ 5. Technical Stack

| Component | Technology |
| :--- | :--- |
| **RL Framework** | Stable-Baselines3 (SB3) |
| **Deep Learning** | PyTorch 2.2+ |
| **LLM Library** | Hugging Face Transformers / Accelerate |
| **Training UI** | W&B (Weights & Biases) |
| **Dashboard** | Gradio 6.0+ |
| **Inference** | BitsAndBytes (4-bit quantization) |
| **Evaluation** | custom `quality_eval_fn` per task type |

---

## 🖥️ 6. Interaction Modes

### **Technical Dashboard (`gradio_app.py`)**
Designed for developers to debug the agent's logic.
- **Worker Fleet Map**: Visualizes which workers are idle, active, or currently being corrected.
- **Thought Stream**: Prints the internal reasoning of the PPO model (e.g., *"Worker 2 has failed 3 times, replacing now..."*).

### **Conversational Chat (`gradio_chat.py`)**
Designed for end-users to experience the power of orchestration.
- **Dual-Response Selection**: For high-uncertainty tasks, the Manager triggers a "Comparison Mode" where the user chooses between two generated options, similar to RLHF (Reinforcement Learning from Human Feedback).

---

## 📁 7. Repository Structure

- `agents/`: Manager and Worker agent definitions.
- `env/`: The Gymnasium environment logic and Task Library.
- `training/`: PPO training scripts, inference wrappers, and SFT logic.
- `models/`: Pre-trained RL policies and worker checkpoints.
- `tests/`: Automated suite for verifying environment integrity.

---
*Document Version: 2.0.0 | Status: Phase 2 Complete*
