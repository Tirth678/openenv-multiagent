# 🎭 OrchestraAI: Multi-Agent RL Orchestration Framework

**OrchestraAI** is an advanced platform for training and evaluating Reinforcement Learning (RL) agents designed to manage multiple LLM "Workers." It solves the orchestration problem: How can a manager maximize quality while staying within a token budget when workers are unpredictable?

## 🏗️ Architecture

```text
    +-------------------+
    |   Manager Agent   | <--- PPO / Heuristic / Parallel
    +---------+---------+
              |
      [Actions: Assign, Check, Approve, Replace]
              |
    +---------v---------+
    |   OpenEnv (Gym)   | <--- Budget & Task Tracking
    +---------+---------+
              |
      [Workers: SmolLM, Llama-3, etc.]
              |
    +---------v---------+
    | Hallucination Eng | <--- Simulated Failures
    +-------------------+
```

## 🚀 Getting Started

### 1. Installation
```bash
git clone https://github.com/Tirth678/openenv-multiagent
cd orchestraai
pip install -r requirements.txt
```

### 2. Run Tests
Ensure the environment is working correctly:
```bash
pytest tests/
```

### 3. Training (Phase 1)
Train the PPO Manager in a simulated environment:
```bash
python training/train_manager.py
```

### 4. Training (Phase 2 - SFT)
Fine-tune worker models to exhibit realistic failure modes:
```bash
python training/sft_workers.py
```

### 5. Launch Dashboards
- **Technical Dashboard**: `python gradio_app.py` (Port 7861)
- **Conversational Chat**: `python gradio_chat.py` (Port 7862)

## ⚙️ Configuration
Modify `config.yaml` to change:
- `num_workers`
- `token_budget`
- `reward` weights
- `worker` skill levels and model IDs

## 📈 Monitoring
OrchestraAI integrates with **Weights & Biases (W&B)**. To see training logs:
1. Run `wandb login`
2. Start training.
3. View your dashboard at `wandb.ai`.

---
*Developed for research into reliable multi-agent systems.*
