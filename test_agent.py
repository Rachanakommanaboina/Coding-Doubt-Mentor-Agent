"""
test_agent.py
-------------

Unit tests for rag_agent.py.

Run with:

    python -m pytest test_agent.py -v

or:

    python test_agent.py
"""

import os
import unittest

from rag_agent import (
    CodingMentorAgent,
    chunk_text,
    load_text_from_file,
)


# ============================================================
# KNOWLEDGE BASE
# ============================================================

SAMPLE_KB_PATH = os.path.join(
    os.path.dirname(__file__),
    "coding_info.txt"
)


# ============================================================
# CHUNKING TESTS
# ============================================================

class TestChunking(unittest.TestCase):

    def test_chunk_text_splits_on_blank_lines(self):
        """
        Check whether paragraphs separated by blank lines
        are converted into separate chunks.
        """

        text = (
            "Topic one line.\n\n"
            "Topic two line.\n\n"
            "Topic three line."
        )

        chunks = chunk_text(text)

        self.assertEqual(
            len(chunks),
            3
        )


    def test_chunk_text_handles_empty_string(self):
        """
        Check that an empty knowledge base does not crash.
        """

        chunks = chunk_text("")

        self.assertTrue(
            len(chunks) >= 1
        )


    def test_chunk_text_splits_long_blocks(self):
        """
        Check that long blocks are divided
        according to max_words.
        """

        long_block = " ".join(
            ["word"] * 300
        )

        chunks = chunk_text(
            long_block,
            max_words=80
        )

        self.assertGreater(
            len(chunks),
            1
        )

        for chunk in chunks:

            self.assertLessEqual(
                len(chunk.split()),
                80
            )


    def test_chunk_text_preserves_content(self):
        """
        Check that important words are preserved
        after chunking.
        """

        text = (
            "Python programming is easy.\n\n"
            "Java is object oriented.\n\n"
            "SQL is used for databases."
        )

        chunks = chunk_text(text)

        combined = " ".join(chunks)

        self.assertIn(
            "Python",
            combined
        )

        self.assertIn(
            "Java",
            combined
        )

        self.assertIn(
            "SQL",
            combined
        )


# ============================================================
# CODING MENTOR AGENT TESTS
# ============================================================

class TestCodingMentorAgent(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """
        Create one CodingMentorAgent for all tests.

        Groq API is disabled during unit testing so that
        the tests remain deterministic and do not consume
        API credits.
        """

        # Save existing Groq API key
        cls._original_groq_key = os.environ.pop(
            "GROQ_API_KEY",
            None
        )

        # Load knowledge base
        if not os.path.exists(
            SAMPLE_KB_PATH
        ):

            raise FileNotFoundError(
                "coding_info.txt was not found.\n"
                "Make sure coding_info.txt is in the "
                "same folder as test_agent.py."
            )

        kb_text = load_text_from_file(
            SAMPLE_KB_PATH
        )

        # Create agent
        cls.agent = CodingMentorAgent(
            kb_text
        )


    @classmethod
    def tearDownClass(cls):
        """
        Restore Groq API key after tests.
        """

        if cls._original_groq_key:

            os.environ[
                "GROQ_API_KEY"
            ] = cls._original_groq_key


    # --------------------------------------------------------
    # AGENT INITIALIZATION
    # --------------------------------------------------------

    def test_agent_indexes_chunks(self):
        """
        Check whether the knowledge base
        was successfully indexed.
        """

        self.assertGreater(
            len(self.agent.chunks),
            0
        )


    def test_agent_has_tfidf_matrix(self):
        """
        Check whether TF-IDF document matrix
        was created.
        """

        self.assertIsNotNone(
            self.agent.doc_matrix
        )


    def test_agent_has_vectorizer(self):
        """
        Check whether TF-IDF vectorizer
        was created.
        """

        self.assertIsNotNone(
            self.agent.vectorizer
        )


    # --------------------------------------------------------
    # RETRIEVAL TESTS
    # --------------------------------------------------------

    def test_retrieve_returns_results(self):
        """
        Check that retrieval returns results
        for a relevant query.
        """

        results = self.agent.retrieve(
            "Python programming",
            top_k=3
        )

        self.assertIsInstance(
            results,
            list
        )


    def test_retrieve_returns_relevant_chunk_for_list_vs_tuple(self):
        """
        Check retrieval for a Python list/tuple question.
        """

        results = self.agent.retrieve(
            "difference between list and tuple in Python",
            top_k=3
        )

        self.assertGreater(
            len(results),
            0
        )

        combined = " ".join(
            chunk
            for chunk, score in results
        ).lower()

        self.assertIn(
            "tuple",
            combined
        )


    def test_retrieve_returns_scores(self):
        """
        Check that retrieval results contain
        numeric similarity scores.
        """

        results = self.agent.retrieve(
            "Python",
            top_k=3
        )

        for chunk, score in results:

            self.assertIsInstance(
                chunk,
                str
            )

            self.assertIsInstance(
                score,
                float
            )


    def test_retrieve_top_k(self):
        """
        Check that top_k limits the number of results.
        """

        results = self.agent.retrieve(
            "programming",
            top_k=2
        )

        self.assertLessEqual(
            len(results),
            2
        )


    def test_retrieve_irrelevant_query_returns_low_or_no_results(self):
        """
        An unrelated query should normally produce
        no strong matches.
        """

        results = self.agent.retrieve(
            "ancient roman history and architecture",
            top_k=3
        )

        for _, score in results:

            self.assertLess(
                score,
                0.3
            )


    # --------------------------------------------------------
    # ANSWER DOUBT TESTS
    # --------------------------------------------------------

    def test_answer_doubt_fallback_returns_text(self):
        """
        Check that answer_doubt() returns text
        even when Groq is disabled.
        """

        answer = self.agent.answer_doubt(
            "What is a hash map used for?"
        )

        self.assertIsInstance(
            answer,
            str
        )

        self.assertGreater(
            len(answer),
            0
        )


    def test_answer_doubt_returns_non_empty_response(self):
        """
        Check that a normal coding question
        produces a non-empty answer.
        """

        answer = self.agent.answer_doubt(
            "What is a variable in Python?"
        )

        self.assertIsInstance(
            answer,
            str
        )

        self.assertGreater(
            len(answer.strip()),
            0
        )


    # --------------------------------------------------------
    # CODE EXPLANATION TESTS
    # --------------------------------------------------------

    def test_explain_code_fallback_returns_text(self):
        """
        Check that explain_code() returns
        a response without Groq.
        """

        snippet = """
def add(a, b):
    return a + b
"""

        explanation = self.agent.explain_code(
            snippet
        )

        self.assertIsInstance(
            explanation,
            str
        )

        self.assertGreater(
            len(explanation),
            0
        )


    def test_explain_code_empty_input(self):
        """
        Check empty code input.
        """

        explanation = self.agent.explain_code(
            ""
        )

        self.assertIsInstance(
            explanation,
            str
        )

        self.assertGreater(
            len(explanation),
            0
        )


    # --------------------------------------------------------
    # ERROR DIAGNOSIS TESTS
    # --------------------------------------------------------

    def test_diagnose_error_detects_known_python_error(self):
        """
        Check IndexError detection.
        """

        diagnosis = self.agent.diagnose_error(
            "IndexError: list index out of range"
        )

        self.assertIsInstance(
            diagnosis,
            str
        )

        self.assertIn(
            "IndexError",
            diagnosis
        )


    def test_diagnose_error_detects_known_java_error(self):
        """
        Check Java NullPointerException detection.
        """

        diagnosis = self.agent.diagnose_error(
            "Exception in thread main "
            "java.lang.NullPointerException"
        )

        self.assertIsInstance(
            diagnosis,
            str
        )

        self.assertIn(
            "NullPointerException",
            diagnosis
        )


    def test_diagnose_error_detects_type_error(self):
        """
        Check TypeError detection.
        """

        diagnosis = self.agent.diagnose_error(
            "TypeError: unsupported operand type(s)"
        )

        self.assertIsInstance(
            diagnosis,
            str
        )

        self.assertIn(
            "TypeError",
            diagnosis
        )


    def test_diagnose_error_detects_name_error(self):
        """
        Check NameError detection.
        """

        diagnosis = self.agent.diagnose_error(
            "NameError: name 'x' is not defined"
        )

        self.assertIsInstance(
            diagnosis,
            str
        )

        self.assertIn(
            "NameError",
            diagnosis
        )


    def test_diagnose_error_unknown_pattern_returns_fallback_text(self):
        """
        Check unknown error handling.
        """

        diagnosis = self.agent.diagnose_error(
            "some totally unrecognized custom error XYZ123"
        )

        self.assertIsInstance(
            diagnosis,
            str
        )

        self.assertGreater(
            len(diagnosis),
            0
        )


    # --------------------------------------------------------
    # KEYWORD TESTS
    # --------------------------------------------------------

    def test_extract_keywords_returns_terms(self):
        """
        Check TF-IDF keyword extraction.
        """

        keywords = self.agent._extract_keywords(
            top_n=5
        )

        self.assertEqual(
            len(keywords),
            5
        )

        for keyword in keywords:

            self.assertIsInstance(
                keyword,
                str
            )


# ============================================================
# RUN TESTS DIRECTLY
# ============================================================

if __name__ == "__main__":

    unittest.main()