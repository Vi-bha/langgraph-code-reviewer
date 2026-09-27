import os
import time
import streamlit as st

from graph_agent import (
    run_review,
    get_mermaid_png_bytes,
    InputValidationError,
    ConfigError,
    MAX_CODE_CHARS,
)

st.set_page_config(
    page_title="LangGraph Code Review Agent",
    page_icon="🧩",
    layout="wide",
)

# ---------- Sidebar ----------
with st.sidebar:
    st.title("🧩 LangGraph Code Reviewer")
    st.caption("Parallel multi-agent PR review, built on LangGraph.")

    st.markdown("---")
    st.markdown("**How it works**")
    st.markdown(
        "Your code is sent to three specialist agents **in parallel** — "
        "security, style, logic — then a tech-lead agent synthesizes their "
        "findings into one review."
    )

    st.markdown("---")
    groq_key_present = bool(os.environ.get("GROQ_API_KEY"))
    if groq_key_present:
        st.success("Groq API key detected")
    else:
        st.error("GROQ_API_KEY not set — add it in Secrets/Environment")

    st.markdown("---")
    show_graph = st.checkbox("Show live graph structure", value=True)
    show_individual = st.checkbox("Show individual agent findings", value=True)

# ---------- Header ----------
st.markdown("## Autonomous Code Review Agent")
st.markdown(
    "Paste a code snippet or load the demo example, then run a full "
    "multi-agent review."
)

if show_graph:
    with st.expander("Graph structure (LangGraph)", expanded=False):
        try:
            png = get_mermaid_png_bytes()
            st.image(png, caption="security / style / logic run in parallel, then aggregate")
        except Exception:
            st.info("Diagram renders once GROQ_API_KEY is set and the graph compiles.")

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

col1, col2 = st.columns([1, 1])
with col1:
    filename = st.text_input("Filename", value="app_module.py")
    if st.button("Load demo snippet"):
        st.session_state["code_input"] = DEMO_CODE

    code_input = st.text_area(
        "Code to review",
        value=st.session_state.get("code_input", ""),
        height=380,
        placeholder="Paste a function, module, or PR diff snippet here...",
    )
    st.caption(f"{len(code_input)} / {MAX_CODE_CHARS} chars")

    run_clicked = st.button("Run review", type="primary", disabled=not code_input.strip())

with col2:
    if run_clicked:
        if not groq_key_present:
            st.error("Set GROQ_API_KEY before running a review.")
        else:
            try:
                start = time.time()
                with st.spinner("Running security, style, and logic agents in parallel..."):
                    result = run_review(code_input, filename=filename)
                elapsed = time.time() - start

                st.success(f"Review complete in {elapsed:.1f}s")
                st.markdown("### Final Review")
                st.markdown(result["final_review"])

                if show_individual:
                    st.markdown("---")
                    st.markdown("### Individual Agent Findings")
                    tabs = st.tabs([f["category"].capitalize() for f in result["findings"]])
                    for tab, finding in zip(tabs, result["findings"]):
                        with tab:
                            st.markdown(finding["content"])

            except InputValidationError as e:
                st.error(f"Invalid input: {e}")
            except ConfigError as e:
                st.error(f"Configuration error: {e}")
            except Exception as e:
                st.error(f"Review failed unexpectedly: {e}")
                st.caption("Check the app logs for details.")
    else:
        st.info("Results will appear here after you run a review.")

st.markdown("---")
st.caption(
    "Built with LangGraph (fan-out/fan-in graph) + Groq (Llama 3.3 70B) + Streamlit. "
    "Free stack, no paid services required."
)
