import gradio as gr
import time
import json
from training.inference import InferencePipeline

# Conversational Chat Logic
class OrchestraChat:
    def __init__(self):
        self.pipe = InferencePipeline(strategy="Parallel")
        self.log_file = "rlhf_preferences.jsonl"
        
    def chat_response(self, message, history):
        history = history or []
        history.append({"role": "user", "content": message})
        yield history, self.get_status()
        
        history.append({"role": "assistant", "content": f"🚀 Received task: '{message}'. Coordinating workers..."})
        yield history, self.get_status()
        
        if "compare" in message.lower() or "website" in message.lower() or "aryan" in message.lower():
            time.sleep(1)
            history.append({"role": "assistant", "content": "🤔 Uncertainty detected in subtask 'Drafting'. Please choose the best response:"})
            history.append({"role": "assistant", "content": "🅰 **Worker A (SmolLM-360M):** The website should be modern and blue.\n\n🅱 **Worker B (Llama-3B):** The website should use a sleek dark theme with neon accents."})
            yield history, self.get_status()
        else:
            for update in self.pipe.stream_episode():
                if update['done']: break
                history.append({"role": "assistant", "content": f"🧠 **Manager:** {update['thought']}"})
                yield history, self.get_status(update)
                time.sleep(0.5)
            
            history.append({"role": "assistant", "content": "✅ Task completed successfully!"})
            yield history, self.get_status()

    def get_status(self, update=None):
        if update:
            return {
                "active_worker": "Worker " + str(update.get('worker_id', 0)),
                "quality_score": 0.85,
                "tokens_used": 5000 - update['budget_remaining'],
                "budget_remaining": update['budget_remaining'],
                "manager_confidence": "HIGH"
            }
        return {"active_worker": "None", "quality_score": 0.0, "tokens_used": 0, "budget_remaining": 5000}

    def select_choice(self, idx, history):
        history.append({"role": "user", "content": f"I'll go with Option {'A' if idx == 0 else 'B'}."})
        history.append({"role": "assistant", "content": f"Integrated successfully. Final task complete! ✅"})
        return history, gr.update(visible=False)

    def log_preference(self, choice):
        with open(self.log_file, "a") as f:
            f.write(json.dumps({"timestamp": time.time(), "choice": choice}) + "\n")
        return f"Logged preference for Option {choice}"

def render_chat_ui():
    chat_logic = OrchestraChat()
    
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500)
            msg = gr.Textbox(placeholder="What should we build today?", label="Task Input")
            with gr.Row():
                submit = gr.Button("Send", variant="primary")
                clear = gr.ClearButton([msg, chatbot])
            
            with gr.Row(visible=True) as choice_row:
                vote_a = gr.Button("🅰 Vote for A")
                vote_b = gr.Button("🅱 Vote for B")
                vote_result = gr.Markdown("")

        with gr.Column(scale=1):
            gr.Markdown("### 📡 Live Status")
            status_json = gr.JSON(value=chat_logic.get_status())
            
            with gr.Accordion("RLHF Preferences Log", open=False):
                log_display = gr.Textbox(label="Recent Logs", lines=5)

    # Event Handlers
    submit.click(chat_logic.chat_response, [msg, chatbot], [chatbot, status_json])
    msg.submit(chat_logic.chat_response, [msg, chatbot], [chatbot, status_json])
    vote_a.click(lambda: chat_logic.log_preference("A"), outputs=vote_result)
    vote_b.click(lambda: chat_logic.log_preference("B"), outputs=vote_result)

def build_chat():
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        render_chat_ui()
    return demo

if __name__ == "__main__":
    demo = build_chat()
    demo.launch(server_name="0.0.0.0", server_port=7862, share=True)
