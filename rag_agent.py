"""
rag_agent.py
------------
Core RAG (Retrieval-Augmented Generation) logic
for the Coding Doubt Mentor Agent.

Features:
    - TF-IDF retrieval
    - Cosine similarity
    - Groq API
    - Coding doubt answering
    - Code explanation
    - Error diagnosis
    - PDF/TXT knowledge-base loading
"""

import io
import os
import re
from typing import List, Tuple, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv


# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()


# =========================================================
# PDF SUPPORT
# =========================================================

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None


# =========================================================
# GROQ SUPPORT
# =========================================================

try:
    from groq import Groq

    _GROQ_AVAILABLE = True

except ImportError:

    Groq = None
    _GROQ_AVAILABLE = False


# =========================================================
# LOAD TEXT FROM FILE
# =========================================================

def load_text_from_file(file_path: str) -> str:
    """
    Load text from a TXT or PDF file.
    """

    if file_path.lower().endswith(".pdf"):

        if PdfReader is None:

            raise RuntimeError(
                "pypdf is not installed. "
                "Run: pip install pypdf"
            )

        reader = PdfReader(file_path)

        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as f:

        return f.read()


# =========================================================
# LOAD TEXT FROM UPLOADED BYTES
# =========================================================

def load_text_from_bytes(
    file_bytes: bytes,
    filename: str
) -> str:
    """
    Load text from uploaded TXT or PDF bytes.
    """

    if filename.lower().endswith(".pdf"):

        if PdfReader is None:

            raise RuntimeError(
                "pypdf is not installed. "
                "Run: pip install pypdf"
            )

        reader = PdfReader(
            io.BytesIO(file_bytes)
        )

        return "\n".join(
            page.extract_text() or ""
            for page in reader.pages
        )

    return file_bytes.decode(
        "utf-8",
        errors="ignore"
    )


# =========================================================
# CHUNK TEXT
# =========================================================

def chunk_text(
    text: str,
    max_words: int = 80
) -> List[str]:
    """
    Split knowledge-base text into
    smaller retrievable chunks.
    """

    if not text or not text.strip():

        return [""]

    raw_blocks = re.split(
        r"\n\s*\n",
        text.strip()
    )

    chunks = []

    for block in raw_blocks:

        block = block.strip()

        if not block:
            continue

        words = block.split()

        if len(words) <= max_words:

            chunks.append(block)

        else:

            for i in range(
                0,
                len(words),
                max_words
            ):

                chunks.append(
                    " ".join(
                        words[
                            i:i + max_words
                        ]
                    )
                )

    return (
        chunks
        if chunks
        else [text.strip()]
    )


# =========================================================
# CODING MENTOR AGENT
# =========================================================

class CodingMentorAgent:
    """
    RAG-based Coding Mentor Agent.

    Uses:
        TF-IDF + cosine similarity
        Groq LLM
    """

    # -----------------------------------------------------
    # INITIALIZATION
    # -----------------------------------------------------

    def __init__(
        self,
        knowledge_text: str
    ):

        self.knowledge_text = (
            knowledge_text
        )

        # ---------------------------------------------
        # CREATE KNOWLEDGE CHUNKS
        # ---------------------------------------------

        self.chunks: List[str] = chunk_text(
            knowledge_text
        )

        # ---------------------------------------------
        # BUILD TF-IDF INDEX
        # ---------------------------------------------

        self.vectorizer = TfidfVectorizer(
            stop_words="english"
        )

        self.doc_matrix = (
            self.vectorizer.fit_transform(
                self.chunks
            )
        )

        # ---------------------------------------------
        # INITIALIZE GROQ
        # ---------------------------------------------

        self.client = None

        api_key = os.environ.get(
            "GROQ_API_KEY",
            ""
        ).strip()

        if _GROQ_AVAILABLE and api_key:

            try:

                self.client = Groq(
                    api_key=api_key
                )

                print(
                    "Groq client initialized successfully."
                )

            except Exception as e:

                print(
                    "Groq client initialization error:",
                    e
                )

                self.client = None

        else:

            if not _GROQ_AVAILABLE:

                print(
                    "Groq package is not installed."
                )

            elif not api_key:

                print(
                    "GROQ_API_KEY was not found."
                )

    # =====================================================
    # RETRIEVAL
    # =====================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Tuple[str, float]]:
        """
        Return the most relevant
        knowledge-base chunks.
        """

        if not query or not query.strip():

            return []

        query_vec = (
            self.vectorizer.transform(
                [query]
            )
        )

        scores = cosine_similarity(
            query_vec,
            self.doc_matrix
        ).flatten()

        top_idx = np.argsort(
            scores
        )[::-1][:top_k]

        return [
            (
                self.chunks[i],
                float(scores[i])
            )

            for i in top_idx

            if scores[i] > 0
        ]

    # =====================================================
    # GROQ LLM CALL
    # =====================================================

    def _call_llm(
        self,
        system: str,
        user: str
    ) -> Optional[str]:
        """
        Call Groq API.

        The model is read from:

            LLM_MODEL

        in the .env file.

        Default:

            openai/gpt-oss-20b
        """

        # ---------------------------------------------
        # CHECK CLIENT
        # ---------------------------------------------

        if not self.client:

            print(
                "Groq client is not initialized."
            )

            return None

        # ---------------------------------------------
        # GET MODEL
        # ---------------------------------------------

        model_name = os.environ.get(
            "LLM_MODEL",
            "openai/gpt-oss-20b"
        ).strip()

        print(
            f"Using Groq model: {model_name}"
        )

        # ---------------------------------------------
        # CALL GROQ
        # ---------------------------------------------

        try:

            response = (
                self.client
                .chat
                .completions
                .create(
                    model=model_name,

                    messages=[
                        {
                            "role": "system",
                            "content": system
                        },
                        {
                            "role": "user",
                            "content": user
                        }
                    ],

                    temperature=0.2,

                    max_tokens=1500
                )
            )

            # -----------------------------------------
            # EXTRACT RESPONSE
            # -----------------------------------------

            if (
                response.choices
                and
                response.choices[0].message
                and
                response.choices[0].message.content
            ):

                return (
                    response
                    .choices[0]
                    .message
                    .content
                )

            print(
                "Groq returned an empty response."
            )

            return None

        # ---------------------------------------------
        # ERROR HANDLING
        # ---------------------------------------------

        except Exception as e:

            print(
                f"Groq API error: "
                f"{type(e).__name__}: {e}"
            )

            return None

    # =====================================================
    # ANSWER CODING DOUBT
    # =====================================================

    def answer_doubt(
        self,
        question: str
    ) -> str:
        """
        Answer a student's coding question
        using RAG + Groq.
        """

        if (
            not question
            or not question.strip()
        ):

            return (
                "Please enter a coding question."
            )

        # ---------------------------------------------
        # RETRIEVE CONTEXT
        # ---------------------------------------------

        top_chunks = self.retrieve(
            question,
            top_k=3
        )

        context = (
            "\n\n".join(
                c
                for c, _ in top_chunks
            )
            or self.knowledge_text[:3000]
        )

        # ---------------------------------------------
        # SYSTEM PROMPT
        # ---------------------------------------------

        system_prompt = (

            "You are a patient and expert "
            "coding mentor.\n\n"

            "Answer the student's coding question "
            "clearly and concisely.\n\n"

            "Rules:\n"

            "1. Use simple student-friendly language.\n"

            "2. Explain concepts step by step.\n"

            "3. Use the supplied reference "
            "context when relevant.\n"

            "4. Give a short code example "
            "when useful.\n"

            "5. Do not invent information "
            "from the reference context.\n"

            "6. If the question is about programming, "
            "provide a practical answer.\n"

            "7. Use Markdown formatting "
            "where appropriate.\n"

            "8. If unrelated to coding, politely "
            "explain that you specialize in coding, "
            "programming, DSA, SQL, Git and OOP."
        )

        # ---------------------------------------------
        # USER PROMPT
        # ---------------------------------------------

        user_prompt = (

            "Reference context:\n\n"

            + context

            + "\n\nStudent's coding doubt:\n\n"

            + question

            + "\n\nProvide a clear and useful answer."
        )

        # ---------------------------------------------
        # GROQ
        # ---------------------------------------------

        llm_answer = self._call_llm(
            system_prompt,
            user_prompt
        )

        if llm_answer:

            return llm_answer

        # ---------------------------------------------
        # FALLBACK
        # ---------------------------------------------

        if top_chunks:

            return (
                "### Based on the knowledge base\n\n"
                + top_chunks[0][0]
            )

        return (
            "I don't have specific reference "
            "material on that yet. "
            "Try rephrasing your question."
        )

    # =====================================================
    # EXPLAIN CODE
    # =====================================================

    def explain_code(
        self,
        code_snippet: str
    ) -> str:
        """
        Explain a pasted code snippet
        using Groq.
        """

        if (
            not code_snippet
            or not code_snippet.strip()
        ):

            return (
                "Please paste a code snippet."
            )

        # ---------------------------------------------
        # SYSTEM PROMPT
        # ---------------------------------------------

        system_prompt = (

            "You are an expert coding mentor.\n\n"

            "Explain the given code snippet clearly "
            "for a student.\n\n"

            "Include:\n"

            "1. Overall purpose\n"

            "2. Important lines\n"

            "3. Variables\n"

            "4. Functions\n"

            "5. Control flow\n"

            "6. Expected output\n"

            "7. Bugs or errors, if present\n"

            "8. Bad practices, if present\n"

            "9. Suggestions for improvement\n\n"

            "Use simple language and Markdown.\n"

            "Explain the code line by line where useful.\n"

            "Do not unnecessarily rewrite the complete program."
        )

        # ---------------------------------------------
        # USER PROMPT
        # ---------------------------------------------

        user_prompt = (

            "Explain this code:\n\n"

            "```text\n"

            + code_snippet

            + "\n```\n\n"

            "Give a clear and beginner-friendly "
            "explanation."
        )

        # ---------------------------------------------
        # CALL GROQ
        # ---------------------------------------------

        llm_answer = self._call_llm(
            system_prompt,
            user_prompt
        )

        # ---------------------------------------------
        # SUCCESS
        # ---------------------------------------------

        if llm_answer:

            return llm_answer

        # ---------------------------------------------
        # FALLBACK
        # ---------------------------------------------

        lines = [
            line
            for line
            in code_snippet.strip().split("\n")
            if line.strip()
        ]

        notes = [
            f"- Line {i + 1}: `{line.strip()}`"
            for i, line
            in enumerate(lines[:20])
        ]

        return (
            "### Fallback explanation\n\n"

            "Groq is not available, so here is "
            "a basic line-by-line breakdown:\n\n"

            + "\n".join(notes)
        )

    # =====================================================
    # DIAGNOSE ERROR
    # =====================================================

    def diagnose_error(
        self,
        error_text: str
    ) -> str:
        """
        Diagnose an error message or traceback
        using RAG + Groq.
        """

        if (
            not error_text
            or not error_text.strip()
        ):

            return (
                "Please paste an error message "
                "or traceback."
            )

        # ---------------------------------------------
        # RETRIEVE RELEVANT CONTEXT
        # ---------------------------------------------

        top_chunks = self.retrieve(
            error_text,
            top_k=2
        )

        context = "\n\n".join(
            c
            for c, _ in top_chunks
        )

        # ---------------------------------------------
        # SYSTEM PROMPT
        # ---------------------------------------------

        system_prompt = (

            "You are an expert debugging mentor.\n\n"

            "Given an error message or traceback:\n"

            "1. Identify the error type.\n"

            "2. Explain what the error means.\n"

            "3. Explain the likely cause.\n"

            "4. Give a concrete fix.\n"

            "5. Provide corrected code "
            "when appropriate.\n"

            "6. Give prevention tips.\n\n"

            "Be specific, practical and "
            "beginner-friendly.\n"

            "Use Markdown formatting."
        )

        # ---------------------------------------------
        # USER PROMPT
        # ---------------------------------------------

        user_prompt = (

            "Reference context:\n\n"

            + (
                context
                or
                "No matching reference context found."
            )

            + "\n\nError / traceback:\n\n"

            + error_text

            + "\n\nDiagnose this error and "
            "provide a practical solution."
        )

        # ---------------------------------------------
        # CALL GROQ
        # ---------------------------------------------

        llm_answer = self._call_llm(
            system_prompt,
            user_prompt
        )

        # ---------------------------------------------
        # SUCCESS
        # ---------------------------------------------

        if llm_answer:

            return llm_answer

        # ---------------------------------------------
        # FALLBACK ERROR KNOWLEDGE
        # ---------------------------------------------

        known_errors = [

            "IndexError",

            "KeyError",

            "TypeError",

            "NameError",

            "IndentationError",

            "AttributeError",

            "ZeroDivisionError",

            "NullPointerException",

            "ArrayIndexOutOfBoundsException",
        ]

        for err in known_errors:

            if err.lower() in error_text.lower():

                matches = [

                    c

                    for c, _ in self.retrieve(
                        err,
                        top_k=1
                    )
                ]

                if matches:

                    return (
                        f"### Detected **{err}**\n\n"
                        + matches[0]
                    )

                return (
                    f"### Detected **{err}**\n\n"

                    "Please check the line mentioned "
                    "in your traceback."
                )

        # ---------------------------------------------
        # GENERIC FALLBACK
        # ---------------------------------------------

        if top_chunks:

            return (
                "### Closest relevant guidance\n\n"
                + top_chunks[0][0]
            )

        return (
            "Couldn't match this error to a "
            "known pattern. Try pasting the "
            "complete traceback for more context."
        )

    # =====================================================
    # EXTRACT KEYWORDS
    # =====================================================

    def _extract_keywords(
        self,
        top_n: int = 8
    ) -> List[str]:
        """
        Return the most prominent TF-IDF terms.
        """

        scores = np.asarray(
            self.doc_matrix.sum(
                axis=0
            )
        ).flatten()

        terms = np.array(
            self.vectorizer
            .get_feature_names_out()
        )

        top_idx = np.argsort(
            scores
        )[::-1][:top_n]

        return [
            terms[i]
            for i in top_idx
        ]