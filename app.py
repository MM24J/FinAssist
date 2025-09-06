import gradio as gr
import io, contextlib, traceback, os

# Import ask() function from cli.py
from finassist.cli import ask

# Suppress plots in Gradio mode
os.environ["FINASSIST_NO_PLOTS"] = "1"

def handle_message(message, history):
    """Capture ask() output and return it to Gradio chat UI."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            result = ask(message)
        
        # Get any printed output
        printed_output = buf.getvalue().strip()
        
        return result if result else printed_output or "(No output)"
            
    except Exception as e:
        tb = traceback.format_exc(limit=2)
        return f"Error:\n{e}\n\n```\n{tb}\n```"

# Define Gradio chat interface
demo = gr.ChatInterface(
    fn=handle_message,
    type="messages",
    title="FinAssist (demo)",
    description="Ask about budget, investments, or savings tips."
)

if __name__ == "__main__":
    demo.launch()