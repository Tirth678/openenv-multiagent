import gradio as gr
import numpy as np
import time
from env import ManagerWorkerEnv, ManagerAction
from agents.manager_agent import load_manager_agent
import os

# Custom CSS for a premium look
CSS = """
.container { max-width: 1200px; margin: auto; }
.dashboard-text { font-family: 'Courier New', Courier, monospace; background-color: #1e1e1e; color: #00ff00; padding: 15px; border-radius: 8px; line-height: 1.2; }
.worker-card { border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin-bottom: 10px; background-color: #f9f9f9; }
.status-active { color: #ff9800; font-weight: bold; }
.status-idle { color: #4caf50; font-weight: bold; }
.reward-positive { color: #4caf50; font-weight: bold; }
.reward-negative { color: #f44336; font-weight: bold; }
"""

class OrchestraUI:
    def __init__(self):
        self.env = None
        self.agent = None
        self.obs = None
        self.done = False
        self.total_reward = 0
        self.step_count = 0
        self.history = []
        self.final_outputs = [] # List of (subtask, output) tuples
        self.all_outputs = [] # List of all generated outputs (including rejected ones)
        self.choice_mode = False
        self.choice_data = None # (subtask_id, worker_a_id, worker_b_id)
        self.option_a_text = ""
        self.option_b_text = ""

    def reset_env(self, max_workers, budget, difficulty, agent_type, custom_task_desc=None, custom_subtasks=None):
        config = {
            'max_workers': int(max_workers),
            'token_budget': int(budget),
            'task_difficulty': int(difficulty),
            'max_steps': 50,
            'failure_injection_rate': 0.6
        }
        self.env = ManagerWorkerEnv(config)
        
        # Handle custom task if provided
        if custom_task_desc and custom_task_desc.strip():
            from env.task_library import Task, Subtask
            
            subtasks_list = []
            if custom_subtasks and custom_subtasks.strip():
                for i, s in enumerate(custom_subtasks.split(',')):
                    subtasks_list.append(Subtask(i, s.strip(), "Text", 0.7))
            else:
                # Default subtasks if none provided
                subtasks_list = [Subtask(0, "Process input", "Text", 0.7)]
                
            custom_task = Task(
                task_id="custom_task",
                task_type="custom",
                description=custom_task_desc,
                subtasks=subtasks_list,
                difficulty=int(difficulty),
                quality_eval_fn=lambda x: 0.8, # Simple default eval
                estimated_tokens=500
            )
            
            # Manually inject task into env (bypassing library sample)
            self.obs = self.env.reset() # Standard reset first
            self.env._current_subtasks = subtasks_list
            self.env.state.task = {
                'task_id': custom_task.task_id,
                'task_type': custom_task.task_type,
                'description': custom_task.description,
                'difficulty': custom_task.difficulty,
                'estimated_tokens': custom_task.estimated_tokens,
                'num_subtasks': len(subtasks_list),
            }
            self.env.state.subtask_status = [False] * len(subtasks_list)
            self.env.state.subtask_assignments = [None] * len(subtasks_list)
            self.obs = self.env._generate_observation()
        else:
            self.obs = self.env.reset()
        if agent_type == "Heuristic":
            self.agent = load_manager_agent(parallel=False)
        elif agent_type == "Parallel":
            self.agent = load_manager_agent(parallel=True)
        else:
            # Try to load a trained model if it exists
            model_path = "models/test_ppo_manager"
            if os.path.exists(model_path + ".zip"):
                self.agent = load_manager_agent(model_path=model_path)
            else:
                self.agent = load_manager_agent(parallel=True)
        
        self.done = False
        self.total_reward = 0
        self.step_count = 0
        self.history = [["Env Reset", "Initial state", 0.0, "Started"]]
        self.final_outputs = []
        self.all_outputs = []
        
        return self.get_ui_update()

    def step(self):
        if self.env is None or self.done:
            return self.get_ui_update()

        action = self.agent.predict(self.obs)
        self.obs, reward, self.done, info = self.env.step(action)
        
        self.total_reward += reward
        self.step_count += 1
        
        action_name = self.env.ACTION_NAMES[action.action_id]
        target = f"W{action.target_worker_id}" if action.target_worker_id is not None else "-"
        
        # Capture all generation events (assign or correct)
        if action.action_id in [0, 2] and action.target_worker_id is not None:
            worker = self.env.state.workers[action.target_worker_id]
            # We check AFTER the step in the next update, but we can also check the buffer now if it's updated
            # Actually, _run_worker is called inside _action_assign_subtask/correct_worker
            # So worker.output_buffer is already populated
            subtask = self.env._get_subtask(worker.current_subtask_id)
            self.all_outputs.insert(0, [f"Step {self.step_count}", f"Worker {worker.worker_id}", subtask.description, worker.output_buffer])

        # If action was 'approve_output', capture the output
        if action.action_id == 5 and action.target_worker_id is not None:
            worker = self.env.state.workers[action.target_worker_id]
            if worker.current_subtask_id is not None:
                subtask = self.env._get_subtask(worker.current_subtask_id)
                self.final_outputs.append([f"Subtask {worker.current_subtask_id}", subtask.description, worker.output_buffer])

        self.history.insert(0, [f"Step {self.step_count}", f"{action_name} ({target})", f"{reward:.2f}", "Done" if self.done else "Active"])
        
        return self.get_ui_update()

    def run_until_done(self):
        outputs = []
        while not self.done and self.step_count < 50:
            yield self.step()
            time.sleep(0.3)

    def get_ui_update(self):
        if self.env is None:
            return "Environment not initialized.", [], "0.00", "0", []

        dashboard = self.env.render(mode='dashboard')
        
        worker_data = []
        for w in self.env.state.workers:
            status = "ACTIVE" if w.is_active else "IDLE"
            worker_data.append([
                f"Worker {w.worker_id}",
                status,
                f"{w.skill_level:.2f}",
                f"{w.progress*100:.1f}%",
                f"{w.output_quality_if_checked:.2f}" if w.is_checked else "?",
                w.failure_mode or "None"
            ])
        
        return (
            dashboard,
            worker_data,
            f"{self.total_reward:.2f}",
            f"{self.step_count}",
            self.history,
            self.agent.last_thought if self.agent else "Waiting...",
            self.final_outputs,
            self.all_outputs,
            self.option_a_text,
            self.option_b_text,
            gr.update(visible=self.choice_mode)
        )

    def start_dual_response(self):
        """Find an unassigned subtask and assign it to two workers for comparison."""
        if self.env is None: return self.get_ui_update()
        
        # Find first unassigned subtask
        target_subtask = None
        for i, assigned in enumerate(self.env.state.subtask_assignments):
            if assigned is None and not self.env.state.subtask_status[i]:
                target_subtask = i
                break
        
        if target_subtask is None:
            self.history.insert(0, ["System", "No unassigned subtasks available for dual response.", 0, "Error"])
            return self.get_ui_update()

        # Pick two idle workers
        idle = [w.worker_id for w in self.env.state.workers if not w.is_active]
        if len(idle) < 2:
            self.history.insert(0, ["System", "Need at least 2 idle workers for dual response.", 0, "Error"])
            return self.get_ui_update()

        w_a, w_b = idle[0], idle[1]
        
        # Run both workers on the same subtask
        subtask_obj = self.env._get_subtask(target_subtask)
        
        # Worker A
        self.env.state.workers[w_a].is_active = True
        self.env.state.workers[w_a].current_subtask_id = target_subtask
        self.env._run_worker(self.env.state.workers[w_a], subtask_obj)
        self.option_a_text = self.env.state.workers[w_a].output_buffer
        
        # Worker B (with slight variation in 'personality' or just second run)
        self.env.state.workers[w_b].is_active = True
        self.env.state.workers[w_b].current_subtask_id = target_subtask
        self.env._run_worker(self.env.state.workers[w_b], subtask_obj, skill_boost=0.1)
        self.option_b_text = self.env.state.workers[w_b].output_buffer
        
        self.choice_mode = True
        self.choice_data = (target_subtask, w_a, w_b)
        self.history.insert(0, ["System", f"Generated dual responses for Subtask {target_subtask}. Please choose.", 0, "Waiting"])
        
        return self.get_ui_update()

    def make_choice(self, choice):
        """User picks Option A or Option B."""
        if not self.choice_mode or not self.choice_data: return self.get_ui_update()
        
        sub_id, w_a, w_b = self.choice_data
        chosen_wid = w_a if choice == "A" else w_b
        other_wid = w_b if choice == "A" else w_a
        
        # Approve the chosen one
        worker = self.env.state.workers[chosen_wid]
        subtask = self.env._get_subtask(sub_id)
        self.final_outputs.append([f"Subtask {sub_id}", subtask.description, worker.output_buffer])
        self.env.state.subtask_status[sub_id] = True
        
        # Free both workers
        for wid in [w_a, w_b]:
            self.env.state.workers[wid].is_active = False
            self.env.state.workers[wid].current_subtask_id = None
            
        self.choice_mode = False
        self.choice_data = None
        self.option_a_text = ""
        self.option_b_text = ""
        self.history.insert(0, ["Human", f"Chose Option {choice} for Subtask {sub_id}.", 10.0, "Approved"])
        
        return self.get_ui_update()

def build_ui():
    ui = OrchestraUI()
    
    with gr.Blocks() as demo:
        gr.Markdown("# 🎭 OrchestraAI: Multi-Agent Orchestration Dashboard")
        gr.Markdown("Visualize how a Manager Agent coordinates a pool of Worker Agents to solve complex tasks.")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Configuration")
                max_workers = gr.Slider(minimum=1, maximum=4, value=4, step=1, label="Max Workers")
                budget = gr.Slider(minimum=500, maximum=5000, value=1000, step=100, label="Token Budget")
                difficulty = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Task Difficulty")
                agent_type = gr.Dropdown(choices=["Heuristic", "Parallel", "PPO Model"], value="Parallel", label="Manager Agent")
                
                reset_btn = gr.Button("🔄 Reset Environment", variant="primary")
                
                with gr.Row():
                    step_btn = gr.Button("⏭️ Single Step")
                    auto_btn = gr.Button("🚀 Run Episode", variant="secondary")
                
                dual_btn = gr.Button("🆚 Dual Response (Choice)", variant="primary")
                
                gr.Markdown("### 📊 Metrics")
                total_reward = gr.Label(value="0.00", label="Total Reward")
                step_counter = gr.Label(value="0", label="Step Count")
                
                gr.Markdown("### 📝 Custom Task (Optional)")
                custom_task_desc = gr.Textbox(label="Task Description", placeholder="e.g. Write a python script for scraping...")
                custom_subtasks = gr.Textbox(label="Subtasks (comma-separated)", placeholder="e.g. Scrape data, Clean data, Save to CSV")
                
                gr.Markdown("### 🧠 Manager Reasoning")
                manager_thought = gr.Textbox(
                    label="Current Thought Process",
                    interactive=False,
                    lines=3,
                    placeholder="Manager thinking..."
                )

            with gr.Column(scale=2):
                gr.Markdown("### 🖥️ System Dashboard")
                dashboard_display = gr.Textbox(
                    label="Real-time Visualization", 
                    lines=15, 
                    max_lines=20, 
                    interactive=False, 
                    elem_classes="dashboard-text",
                    placeholder="Dashboard will appear here..."
                )
                
                gr.Markdown("### 👷 Worker Fleet")
                worker_table = gr.Dataframe(
                    headers=["ID", "Status", "Skill", "Progress", "Quality", "Failure"],
                    datatype=["str", "str", "str", "str", "str", "str"],
                    label="Active Workers"
                )

        with gr.Row():
            gr.Markdown("### 📜 Action History")
            history_table = gr.Dataframe(
                headers=["Step", "Action", "Reward", "Status"],
                datatype=["str", "str", "str", "str"],
                label="Episode Log"
            )

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📄 Approved Outputs")
                final_output_table = gr.Dataframe(
                    headers=["ID", "Subtask", "Content"],
                    datatype=["str", "str", "str"],
                    label="Final Approved Results"
                )
            with gr.Column(scale=1):
                gr.Markdown("### 📡 Worker Output Stream")
                all_output_table = gr.Dataframe(
                    headers=["Step", "Worker", "Subtask", "Content"],
                    datatype=["str", "str", "str", "str"],
                    label="All Generated Text (History)"
                )

        with gr.Column(visible=False) as choice_row:
            gr.Markdown("### 🆚 Dual Response Comparison (Choose the best one)")
            with gr.Row():
                option_a = gr.Textbox(label="Option A", lines=8, interactive=False)
                option_b = gr.Textbox(label="Option B", lines=8, interactive=False)
            with gr.Row():
                choose_a = gr.Button("✅ Choose Option A", variant="primary")
                choose_b = gr.Button("✅ Choose Option B", variant="primary")

        # Event handlers
        reset_btn.click(
            ui.reset_env, 
            inputs=[max_workers, budget, difficulty, agent_type, custom_task_desc, custom_subtasks], 
            outputs=[dashboard_display, worker_table, total_reward, step_counter, history_table, manager_thought, final_output_table, all_output_table]
        )
        
        step_btn.click(
            ui.step, 
            outputs=[dashboard_display, worker_table, total_reward, step_counter, history_table, manager_thought, final_output_table, all_output_table]
        )
        
        auto_btn.click(
            ui.run_until_done, 
            outputs=[dashboard_display, worker_table, total_reward, step_counter, history_table, manager_thought, final_output_table, all_output_table, option_a, option_b, choice_row]
        )

        dual_btn.click(
            ui.start_dual_response,
            outputs=[dashboard_display, worker_table, total_reward, step_counter, history_table, manager_thought, final_output_table, all_output_table, option_a, option_b, choice_row]
        )

        choose_a.click(
            lambda: ui.make_choice("A"),
            outputs=[dashboard_display, worker_table, total_reward, step_counter, history_table, manager_thought, final_output_table, all_output_table, option_a, option_b, choice_row]
        )

        choose_b.click(
            lambda: ui.make_choice("B"),
            outputs=[dashboard_display, worker_table, total_reward, step_counter, history_table, manager_thought, final_output_table, all_output_table, option_a, option_b, choice_row]
        )

        # Initialize
        demo.load(
            ui.reset_env, 
            inputs=[max_workers, budget, difficulty, agent_type, custom_task_desc, custom_subtasks], 
            outputs=[dashboard_display, worker_table, total_reward, step_counter, history_table, manager_thought, final_output_table, all_output_table, option_a, option_b, choice_row]
        )

    return demo

if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7861, css=CSS, theme=gr.themes.Soft(), share=True)
