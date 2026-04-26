import gradio as gr
import numpy as np
import time
import os
from env import ManagerWorkerEnv, ManagerAction
from agents.manager_agent import load_manager_agent

# ChatGPT-inspired CSS
CSS = """
.chat-container { height: 600px; overflow-y: auto; }
.manager-thought { background-color: #f7f7f8; border-left: 4px solid #10a37f; padding: 10px; margin: 10px 0; font-style: italic; }
.dual-choice-container { display: flex; gap: 20px; margin: 20px 0; }
.choice-card { flex: 1; border: 1px solid #e5e5e5; border-radius: 8px; padding: 15px; cursor: pointer; transition: transform 0.2s; }
.choice-card:hover { transform: translateY(-5px); border-color: #10a37f; }
.worker-badge { font-size: 0.8em; background: #eee; padding: 2px 6px; border-radius: 4px; margin-bottom: 8px; display: inline-block; }
"""

class OrchestraChatUI:
    def __init__(self):
        self.env = None
        self.agent = None
        self.obs = None
        self.done = False
        self.choice_mode = False
        self.choice_data = None

    def init_env(self):
        config = {'max_workers': 4, 'token_budget': 2000, 'task_difficulty': 3, 'max_steps': 50}
        self.env = ManagerWorkerEnv(config)
        self.agent = load_manager_agent(parallel=True)

    def handle_user_input(self, user_msg, history):
        if not self.env: self.init_env()
        history = history or []
        history.append([user_msg, None])
        yield history, ""

        history.append([None, f"🚀 Starting task: **{user_msg}**\n\nI'm coordinating 4 worker agents..."])
        yield history, ""

        # Inject custom task
        from env.task_library import Task, Subtask
        subtasks_list = [Subtask(i, f"Task Phase {i+1}", "Text", 0.7) for i in range(3)]
        self.env.reset()
        self.env._current_subtasks = subtasks_list
        self.env.state.task = {'description': user_msg, 'num_subtasks': 3}
        self.env.state.subtask_status = [False] * 3
        self.env.state.subtask_assignments = [None] * 3
        self.obs = self.env._generate_observation()
        
        for _ in range(5):
            if self.done: break
            action = self.agent.predict(self.obs)
            self.obs, r, self.done, _ = self.env.step(action)
            thought = getattr(self.agent, 'last_thought', 'Thinking...')
            history.append([None, f"🧠 *Manager:* {thought}"])
            if action.action_id == 5:
                w = self.env.state.workers[action.target_worker_id]
                history.append([None, f"✅ **Result:**\n\n{w.output_buffer}"])
            time.sleep(0.4)
            yield history, ""

        # Force Dual Response
        history.append([None, "🤔 I have two different versions for the next part. Which one do you prefer?"])
        w_a, w_b = 0, 1
        sub_obj = self.env._get_subtask(2)
        self.env._run_worker(self.env.state.workers[w_a], sub_obj)
        self.env._run_worker(self.env.state.workers[w_b], sub_obj, skill_boost=0.2)
        self.choice_data = (2, w_a, w_b, self.env.state.workers[w_a].output_buffer, self.env.state.workers[w_b].output_buffer)
        
        history.append([None, f"**Option A:** {self.choice_data[3]}\n\n**Option B:** {self.choice_data[4]}"])
        yield history, gr.update(visible=True)

    def select_choice(self, idx, history):
        _, _, _, out_a, out_b = self.choice_data
        chosen = out_a if idx == 0 else out_b
        history.append([f"I'll go with Option {'A' if idx == 0 else 'B'}.", None])
        history.append([None, f"Integrated successfully. Final task complete! ✅\n\n{chosen}"])
        return history, gr.update(visible=False)

def build_chat_ui():
    ui = OrchestraChatUI()
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(height=600)
        with gr.Row(visible=False) as btn_row:
            btn_a = gr.Button("Choose Option A")
            btn_b = gr.Button("Choose Option B")
        with gr.Row():
            txt = gr.Textbox(show_label=False, placeholder="What should we build today?", scale=4)
            sub = gr.Button("Send", scale=1)

        sub.click(ui.handle_user_input, [txt, chatbot], [chatbot, txt, btn_row])
        txt.submit(ui.handle_user_input, [txt, chatbot], [chatbot, txt, btn_row])
        btn_a.click(ui.select_choice, [gr.State(0), chatbot], [chatbot, btn_row])
        btn_b.click(ui.select_choice, [gr.State(1), chatbot], [chatbot, btn_row])
    return demo

if __name__ == "__main__":
    demo = build_chat_ui()
    demo.launch(server_name="0.0.0.0", server_port=7862, css=CSS, share=True)
