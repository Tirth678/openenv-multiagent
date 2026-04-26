import gradio as gr
from gradio_app import render_dashboard_ui
from gradio_chat import render_chat_ui

def build_unified_ui():
    with gr.Blocks(theme=gr.themes.Soft(), title="OrchestraAI Unified Interface") as demo:
        gr.Markdown("# 🎭 OrchestraAI: Unified Multi-Agent Control Center")
        
        with gr.Tabs():
            with gr.Tab("💬 Conversational Chat"):
                render_chat_ui()
                
            with gr.Tab("📊 Technical Dashboard"):
                render_dashboard_ui()
                
        gr.Markdown("---")
        gr.Markdown("*OrchestraAI v2.0 - Powered by Reinforcement Learning & LLMs*")
        
    return demo

if __name__ == "__main__":
    demo = build_unified_ui()
    # Using Port 7860 as the unified entry point
    demo.launch(server_name="0.0.0.0", server_port=7860, share=True)
