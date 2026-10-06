from ollama import chat


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "llama3.2:3b"

FALLBACK = (
    "I couldn't find the answer in the provided documents."
)


# ============================================================
# RESULT HELPERS
# ============================================================

def get_document(item):
    return item[0]


def get_source_info(item):

    doc = get_document(item)

    source = doc.metadata.get(
        "source_file",
        "Unknown"
    )

    page = doc.metadata.get(
        "page_number",
        doc.metadata.get(
            "page",
            "Unknown"
        )
    )

    return source, page


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(results):

    context_parts = []

    for item in results:

        doc = get_document(item)

        content = doc.page_content.strip()

        context_parts.append(
            content
        )

    return "\n\n".join(
        context_parts
    )


# ============================================================
# SOURCE CITATIONS
# ============================================================

def get_sources(results):

    sources = []
    seen = set()

    for item in results:

        source, page = get_source_info(
            item
        )

        key = (
            source,
            page
        )

        if key in seen:
            continue

        seen.add(key)

        sources.append(
            f"[Source: {source}, p.{page}]"
        )

    return sources


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(
    query,
    results
):

    if not results:
        return FALLBACK

    context = build_context(
        results
    )

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY the information
contained in the provided document context.

RULES:

1. Use only information from the document context.
2. Answer the exact question.
3. Do not use outside knowledge.
4. Do not invent facts.
5. Do not combine unrelated information.
6. Prefer the most directly relevant information.
7. If one document chunk directly answers the question,
   prefer that information over loosely related information.
8. Keep simple factual questions concise.
9. For list questions, provide only the relevant items.
10. Never mention SOURCE 1, SOURCE 2, SOURCE 3, etc.
11. Never mention chunks, retrieval, FAISS, embeddings,
    similarity scores, reranking, or internal system details.
12. Never mention how the answer was retrieved.
13. Do not repeat the question.
14. Do not add unrelated examples.
15. If the answer is not supported by the context,
    reply exactly:

I couldn't find the answer in the provided documents.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{query}

ANSWER:
"""

    try:

        response = chat(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            options={
                "temperature": 0,
                "num_predict": 180
            }
        )

        answer = (
            response["message"]["content"]
            .strip()
        )

        if not answer:
            return FALLBACK

        # ----------------------------------------------------
        # Remove accidental source leakage
        # ----------------------------------------------------

        forbidden_patterns = [
            "according to source",
            "source 1",
            "source 2",
            "source 3",
            "source 4",
            "source 5",
        ]

        answer_lower = answer.lower()

        if any(
            phrase in answer_lower
            for phrase in forbidden_patterns
        ):

            # Ask the model one more time with a
            # stricter instruction rather than exposing
            # internal retrieval labels.

            retry_prompt = f"""
Rewrite the following answer.

IMPORTANT:

- Answer only from the provided context.
- Do not mention SOURCE numbers.
- Do not mention documents, chunks, retrieval,
  FAISS, embeddings, or internal system details.
- Answer directly and concisely.
- Do not add new information.

USER QUESTION:
{query}

CONTEXT:
{context}

ANSWER TO REWRITE:
{answer}

REWRITTEN ANSWER:
"""

            retry = chat(
                model=MODEL_NAME,
                messages=[
                    {
                        "role": "user",
                        "content": retry_prompt
                    }
                ],
                options={
                    "temperature": 0,
                    "num_predict": 180
                }
            )

            answer = (
                retry["message"]["content"]
                .strip()
            )

        # ----------------------------------------------------
        # Unsupported-answer normalization
        # ----------------------------------------------------

        unsupported_phrases = [
            "the documents do not explicitly state",
            "the document does not explicitly state",
            "the documents don't explicitly state",
            "the document doesn't explicitly state",
            "the provided documents do not",
            "the provided document does not",
            "not explicitly mentioned in the documents",
            "not mentioned in the documents",
            "cannot be determined from the documents",
            "not enough information in the documents",
        ]

        answer_lower = answer.lower()

        if any(
            phrase in answer_lower
            for phrase in unsupported_phrases
        ):

            return FALLBACK

        return answer

    except Exception:

        return (
            "I couldn't generate an answer right now."
        )


# ============================================================
# FINAL ANSWER + SOURCES
# ============================================================

def generate_answer_with_sources(
    query,
    results
):

    answer = generate_answer(
        query,
        results
    )

    if answer == FALLBACK:
        return answer

    sources = get_sources(
        results
    )

    if not sources:
        return answer

    return (
        answer
        + "\n\n"
        + "\n".join(sources)
    )