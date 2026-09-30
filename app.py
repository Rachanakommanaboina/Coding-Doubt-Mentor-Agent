"""
app.py
------
Streamlit front-end for the Coding Doubt Mentor Agent.

Features:
1. Ask a coding doubt
2. Explain code
3. Diagnose errors
4. Upload PDF/TXT notes
5. Groq AI generation
6. TF-IDF based RAG
"""

import os

import streamlit as st
from dotenv import load_dotenv

from rag_agent import (
    CodingMentorAgent,
    load_text_from_bytes,
    load_text_from_file,
)

# ---------------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# ---------------------------------------------------------

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "openai/gpt-oss-20b"
).strip()

# ---------------------------------------------------------
# SAMPLE KNOWLEDGE BASE
# ---------------------------------------------------------

SAMPLE_KB_PATH = os.path.join(
    os.path.dirname(__file__),
    "coding_info.txt"
)

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="Coding Doubt Mentor Agent",
    page_icon="🧑‍💻",
    layout="wide",
)

# ---------------------------------------------------------
# HEADER
# ---------------------------------------------------------

st.title("🧑‍💻 Coding Doubt Mentor Agent")

st.caption(
    "Ask coding doubts, explain code, diagnose errors, "
    "and get answers grounded in your coding knowledge base."
)

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("⚙️ Configuration")

    # Check Groq API key
    if GROQ_API_KEY:
        st.success("✅ GROQ_API_KEY detected")
    else:
        st.warning(
            "⚠️ GROQ_API_KEY not found. "
            "The application may use fallback mode."
        )

    # Show model
    st.info(
        f"🤖 **Model:** `{LLM_MODEL}`"
    )

    st.divider()

    # Knowledge base
    st.header("📚 Knowledge Base")

    uploaded_file = st.file_uploader(
        "Upload your own notes",
        type=["pdf", "txt"],
        help="Upload a PDF or TXT file containing coding notes."
    )

    use_sample = st.checkbox(
        "Use bundled coding knowledge base",
        value=uploaded_file is None
    )

    st.divider()

    st.info(
        """
        **Supported topics**

        • Python
        • Java
        • DSA
        • SQL
        • Git
        • OOP
        • Debugging
        """
    )

# ---------------------------------------------------------
# LOAD KNOWLEDGE BASE
# ---------------------------------------------------------

kb_text = None
source_label = None

if uploaded_file is not None and not use_sample:

    try:

        kb_text = load_text_from_bytes(
            uploaded_file.read(),
            uploaded_file.name
        )

        source_label = uploaded_file.name

    except Exception as e:

        st.error(
            f"Could not read uploaded file: {e}"
        )

        st.stop()

else:

    try:

        kb_text = load_text_from_file(
            SAMPLE_KB_PATH
        )

        source_label = "coding_info.txt (bundled)"

    except Exception as e:

        st.error(
            f"Could not load coding_info.txt: {e}"
        )

        st.stop()

# ---------------------------------------------------------
# VALIDATE KNOWLEDGE BASE
# ---------------------------------------------------------

if not kb_text or not kb_text.strip():

    st.warning(
        "Couldn't extract any text from the knowledge base. "
        "Please try another file."
    )

    st.stop()

# ---------------------------------------------------------
# BUILD AGENT
# ---------------------------------------------------------

@st.cache_resource(
    show_spinner="Indexing knowledge base..."
)
def build_agent(kb_text: str):

    # IMPORTANT:
    # The current CodingMentorAgent constructor
    # accepts the knowledge base only.
    #
    # Groq configuration is loaded inside rag_agent.py
    # through environment variables.

    return CodingMentorAgent(kb_text)


agent = build_agent(kb_text)

# ---------------------------------------------------------
# AGENT STATUS
# ---------------------------------------------------------

st.caption(
    f"📚 Currently loaded knowledge base: "
    f"**{source_label}** • "
    f"{len(agent.chunks)} indexed sections"
)

# ---------------------------------------------------------
# SIDEBAR AGENT STATUS
# ---------------------------------------------------------

with st.sidebar:

    st.divider()

    if GROQ_API_KEY:

        st.success(
            "🟢 Groq API configured"
        )

    else:

        st.warning(
            "🟡 Groq API not configured"
        )

# ---------------------------------------------------------
# VIEW KNOWLEDGE BASE
# ---------------------------------------------------------

with st.expander("📖 View knowledge base text"):

    st.text(kb_text)

st.divider()

# ---------------------------------------------------------
# TABS
# ---------------------------------------------------------

tab_doubt, tab_explain, tab_error = st.tabs(
    [
        "❓ Ask a coding doubt",
        "📖 Explain my code",
        "🐞 Diagnose an error",
    ]
)

# =========================================================
# TAB 1 — CODING DOUBT
# =========================================================

with tab_doubt:

    st.subheader(
        "❓ Ask a Coding Doubt"
    )

    question = st.text_input(
        "Enter your coding question:",
        placeholder=(
            "Example: What is the difference between "
            "a list and tuple in Python?"
        ),
        key="doubt_input",
    )

    if st.button(
        "🚀 Get Answer",
        key="doubt_button"
    ):

        if not question.strip():

            st.warning(
                "Please enter a coding question."
            )

        else:

            with st.spinner(
                "🤖 Thinking..."
            ):

                try:

                    answer = agent.answer_doubt(
                        question
                    )

                    st.markdown(
                        "### 💡 Answer"
                    )

                    st.write(answer)

                except Exception as e:

                    st.error(
                        f"Error while generating answer: {e}"
                    )

            # Retrieved context
            with st.expander(
                "🔎 Show retrieved reference sections"
            ):

                try:

                    results = agent.retrieve(
                        question,
                        top_k=3
                    )

                    if results:

                        for chunk, score in results:

                            st.markdown(
                                f"**Relevance:** `{score:.2f}`"
                            )

                            st.write(chunk)

                            st.divider()

                    else:

                        st.info(
                            "No strongly matching "
                            "reference section found."
                        )

                except Exception as e:

                    st.warning(
                        f"Could not retrieve references: {e}"
                    )

# =========================================================
# TAB 2 — EXPLAIN CODE
# =========================================================

with tab_explain:

    st.subheader(
        "📖 Explain My Code"
    )

    code_snippet = st.text_area(
        "Paste your code:",
        height=250,
        placeholder="""def add(a, b):
    return a + b

print(add(10, 20))"""
    )

    if st.button(
        "🔍 Explain Code",
        key="explain_button"
    ):

        if not code_snippet.strip():

            st.warning(
                "Please paste some code."
            )

        else:

            with st.spinner(
                "🤖 Analyzing your code..."
            ):

                try:

                    explanation = agent.explain_code(
                        code_snippet
                    )

                    st.markdown(
                        "### 📖 Explanation"
                    )

                    st.write(explanation)

                except Exception as e:

                    st.error(
                        f"Error while explaining code: {e}"
                    )

# =========================================================
# TAB 3 — ERROR DIAGNOSIS
# =========================================================

with tab_error:

    st.subheader(
        "🐞 Diagnose an Error"
    )

    error_text = st.text_area(
        "Paste your error or traceback:",
        height=250,
        placeholder="""IndexError: list index out of range"""
    )

    if st.button(
        "🛠️ Diagnose Error",
        key="error_button"
    ):

        if not error_text.strip():

            st.warning(
                "Please enter an error message "
                "or traceback."
            )

        else:

            with st.spinner(
                "🤖 Diagnosing the error..."
            ):

                try:

                    diagnosis = agent.diagnose_error(
                        error_text
                    )

                    st.markdown(
                        "### 🛠️ Diagnosis & Suggested Fix"
                    )

                    st.write(diagnosis)

                except Exception as e:

                    st.error(
                        f"Error while diagnosing: {e}"
                    )

# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "🧠 Coding Doubt Mentor uses TF-IDF retrieval + "
    "Groq LLM generation. "
    "No code or user data is stored between sessions."
)