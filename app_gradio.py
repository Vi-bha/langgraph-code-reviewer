import os
import time

import gradio as gr

from graph_agent import (
    MAX_CODE_CHARS,
    ConfigError,
    InputValidationError,
    get_mermaid_png_bytes,
    run_review,
)

# ---------- ZeroGPU compatibility shim ----------
# This app is API-only (Groq over HTTP) and never touches a GPU. But on
# Hugging Face's free Gradio tier, hardware defaults to ZeroGPU, which
# refuses to start unless at least one function is decorated with
# @spaces.GPU. This is a documented, standard workaround: a harmless no-op
# function satisfies the startup check without the app ever actually
# requesting GPU time. Falls back to a no-op decorator anywhere else
# (Streamlit Cloud, local dev) where the `spaces` package isn't installed.
try:
    import spaces

    @spaces.GPU
    def _zerogpu_startup_shim():
        """No-op — satisfies HF ZeroGPU's startup check. This app runs
        entirely on CPU; it never issues GPU work."""
        return

    _zerogpu_startup_shim()
except ImportError:
    pass

DEMO_CODE = '''import os

def load_user(user_id):
    query = "SELECT * FROM users WHERE id = " + user_id
    result = db.execute(query)
    return result[0]

def process_items(items):
    total = 0
    for i in range(len(items)):
        total = total + items[i].price
    return total / len(items)

API_KEY = "sk-live-51H8f2example12345"

def load_config(path):
    return eval(open(path).read())
'''


def load_demo():
    return DEMO_CODE, "app_module.py"


def run_review_ui(code: str, filename: str):
    if not code or not code.strip():
        return "Paste some code first.", "", "", ""

    if not os.environ.get("GROQ_API_KEY"):
        return "GROQ_API_KEY is not set for this Space.", "", "", ""

    try:
        start = time.time()
        result = run_review(code, filename=filename or "submitted_file.py")
        elapsed = time.time() - start

        final_md = f"**Review complete in {elapsed:.1f}s**\n\n{result['final_review']}"
        by_category = {f["category"]: f["content"] for f in result["findings"]}
        return (
            final_md,
            by_category.get("security", "_no findings_"),
            by_category.get("style", "_no findings_"),
            by_category.get("logic", "_no findings_"),
        )
    except InputValidationError as e:
        return f"Invalid input: {e}", "", "", ""
    except ConfigError as e:
        return f"Configuration error: {e}", "", "", ""
    except Exception as e:
        return f"Review failed unexpectedly: {e}", "", "", ""


def _load_graph_image():
    """Rendered once at startup. Building the graph doesn't call the LLM,
    so this works even before GROQ_API_KEY is validated."""
    try:
        png_bytes = get_mermaid_png_bytes()
        path = "/tmp/graph_structure.png"
        with open(path, "wb") as f:
            f.write(png_bytes)
        return path
    except Exception:
        return None


_graph_image_path = _load_graph_image()

with gr.Blocks(title="LangGraph Code Review Agent") as demo:
    gr.Markdown("## 🧩 LangGraph Code Reviewer")
    gr.Markdown(
        "Your code is sent to three specialist agents **in parallel** — "
        "security, style, logic — then a tech-lead agent synthesizes their "
        "findings into one review."
    )

    key_status = (
        "✅ **GROQ_API_KEY detected**"
        if os.environ.get("GROQ_API_KEY")
        else "❌ **GROQ_API_KEY not set** — add it in this Space's Settings → Variables and secrets"
    )
    gr.Markdown(key_status)

    with gr.Accordion("Graph structure (LangGraph)", open=False):
        if _graph_image_path:
            gr.Image(
                value=_graph_image_path,
                show_label=False,
                interactive=False,
            )
        else:
            gr.Markdown("_Graph diagram unavailable._")
        gr.Markdown(
            "security / style / logic run in parallel via LangGraph's fan-out, "
            "then aggregate waits for all three (fan-in)."
        )

    with gr.Row():
        with gr.Column():
            filename_input = gr.Textbox(label="Filename", value="app_module.py")
            demo_btn = gr.Button("Load demo snippet")
            code_input = gr.Textbox(
                label=f"Code to review (max {MAX_CODE_CHARS} chars)",
                lines=18,
                placeholder="Paste a function, module, or PR diff snippet here...",
            )
            run_btn = gr.Button("Run review", variant="primary")

        with gr.Column():
            final_output = gr.Markdown(label="Final Review")
            with gr.Tabs():
                with gr.Tab("Security"):
                    security_output = gr.Markdown()
                with gr.Tab("Style"):
                    style_output = gr.Markdown()
                with gr.Tab("Logic"):
                    logic_output = gr.Markdown()

    demo_btn.click(fn=load_demo, outputs=[code_input, filename_input])
    run_btn.click(
        fn=run_review_ui,
        inputs=[code_input, filename_input],
        outputs=[final_output, security_output, style_output, logic_output],
    )

    gr.Markdown("---")
    gr.Markdown(
        "Built with LangGraph (fan-out/fan-in graph) + Groq (GPT-OSS 120B) + Gradio. "
        "Free stack, no paid services required."
    )

if __name__ == "__main__":
    demo.launch()
