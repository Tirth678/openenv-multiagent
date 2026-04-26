import gradio as gr
import pandas as pd
import numpy as np
import time
from training.inference import InferencePipeline
from env.task_library import TaskLibrary

# Technical Dashboard Logic
CSS = """
.dashboard-container { background-color: #f0f2f6; }
.worker-card { border: 1px solid #ddd; padding: 10px; border-radius: 5px; margin: 5px; }
.manager-thought { font-family: monospace; font-size: 0.9em; }
"""

def render_dashboard_ui():
    task_lib = TaskLibrary()
    
    gr.Markdown("### ⚙️ Controls")
    with gr.Row():
        with gr.Column(scale=1):
            strategy_selector = gr.Radio(
                ["PPO", "Heuristic", "Parallel"], 
                value="Parallel", 
                label="Manager Strategy"
            )
            task_selector = gr.Dropdown(
                choices=task_lib.get_all_task_names(),
                value=task_lib.get_all_task_names()[0],
                label="Select Task"
            )
            speed_slider = gr.Slider(0.0, 2.0, value=0.5, label="Step Delay (s)")
            
            with gr.Row():
                run_btn = gr.Button("🚀 Run Episode", variant="primary")
            
            gr.Markdown("### 📊 Metrics")
            reward_label = gr.Label(value="0.00", label="Cumulative Reward")
            budget_label = gr.Label(value="5000", label="Budget Remaining")
            
        with gr.Column(scale=2):
            gr.Markdown("### 🖥️ Worker Fleet & Thought Stream")
            worker_json = gr.JSON(label="Worker States")
            thought_stream = gr.Textbox(
                label="Manager Thought Stream", 
                lines=10, 
                max_lines=15, 
                interactive=False
            )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📜 Action Log")
            action_table = gr.Dataframe(
                headers=["Step", "Action", "Worker", "Subtask", "Reward"],
                datatype=["number", "str", "number", "str", "number"],
                label="Live Execution Log"
            )
        with gr.Column():
            gr.Markdown("### 📈 Reward History")
            reward_plot = gr.LinePlot(
                x="step", 
                y="cumulative_reward", 
                title="Learning Progress"
            )

    def run_episode_stream(strategy, task_name, speed):
        pipe = InferencePipeline(strategy=strategy)
        task_id = next(t['task_id'] for t in task_lib.tasks if t['name'] == task_name)
        
        history = []
        plot_data = []
        thoughts = ""
        cum_reward = 0
        
        for update in pipe.stream_episode(task_id):
            step = update['step']
            action = update['action']
            reward = update['reward']
            cum_reward += reward
            history.insert(0, [step, action, update.get('worker_id', 0), update.get('subtask_id', '-'), reward])
            plot_data.append({"step": step, "cumulative_reward": cum_reward, "budget": update['budget_remaining']})
            thoughts += f"Step {step}: {update['thought']}\n"
            
            yield (
                update['worker_states'],
                thoughts,
                f"{cum_reward:.2f}",
                str(update['budget_remaining']),
                history,
                pd.DataFrame(plot_data)
            )
            time.sleep(speed)

    run_btn.click(
        run_episode_stream,
        inputs=[strategy_selector, task_selector, speed_slider],
        outputs=[worker_json, thought_stream, reward_label, budget_label, action_table, reward_plot]
    )

def build_dashboard():
    with gr.Blocks(theme=gr.themes.Soft(), css=CSS) as demo:
        render_dashboard_ui()
    return demo

if __name__ == "__main__":
    demo = build_dashboard()
    demo.launch(server_name="0.0.0.0", server_port=7861, share=True)
