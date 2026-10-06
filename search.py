import re
import math
from collections import Counter

from vector_store import load_vector_db


# ============================================================
# CONFIG
# ============================================================

INITIAL_K = 8

# For simple factual/list questions
SIMPLE_FINAL_K = 2

# For explanation questions
DETAILED_FINAL_K = 4


# Reranking weights
SEMANTIC_WEIGHT = 0.50
KEYWORD_WEIGHT = 0.25
INTENT_WEIGHT = 0.15
DIRECTNESS_WEIGHT = 0.10


# ============================================================
# LOAD VECTOR DATABASE
# ============================================================

vector_db = load_vector_db()


# ============================================================
# STOP WORDS
# ============================================================

STOP_WORDS = {
    "what",
    "is",
    "are",
    "the",
    "a",
    "an",
    "of",
    "for",
    "to",
    "in",
    "on",
    "and",
    "or",
    "does",
    "do",
    "how",
    "why",
    "can",
    "could",
    "would",
    "should",
    "tell",
    "me",
    "give",
    "please",
}


# ============================================================
# QUERY CLEANING
# ============================================================

def clean_query(query):
    query = query.strip()
    query = re.sub(r"\s+", " ", query)

    return query


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text):
    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower()
    )


def meaningful_tokens(text):
    tokens = tokenize(text)

    return [
        token
        for token in tokens
        if token not in STOP_WORDS
        and len(token) > 1
    ]


# ============================================================
# QUERY INTENT
# ============================================================

def detect_query_intent(query):

    q = query.lower()

    if any(
        phrase in q
        for phrase in [
            "example",
            "examples",
            "application",
            "applications",
            "use case",
            "use cases",
            "uses",
            "types",
            "type",
            "advantages",
            "benefits",
            "disadvantages",
            "limitations",
        ]
    ):
        return "list"

    if any(
        phrase in q
        for phrase in [
            "what is",
            "define",
            "definition",
            "meaning",
        ]
    ):
        return "definition"

    if any(
        phrase in q
        for phrase in [
            "how does",
            "how do",
            "how it works",
            "working",
            "process",
            "steps",
        ]
    ):
        return "explanation"

    if any(
        phrase in q
        for phrase in [
            "why",
            "reason",
        ]
    ):
        return "explanation"

    return "general"


# ============================================================
# QUERY EXPANSION
# ============================================================

def expand_query(query):

    q = query.lower()

    expansions = []

    if any(
        word in q
        for word in [
            "example",
            "examples",
            "instance"
        ]
    ):
        expansions.extend([
            "examples",
            "use cases",
            "applications"
        ])

    if any(
        word in q
        for word in [
            "application",
            "applications",
            "use",
            "uses",
            "used for",
            "use case",
            "use cases"
        ]
    ):
        expansions.extend([
            "examples",
            "applications",
            "uses",
            "use cases"
        ])

    if any(
        word in q
        for word in [
            "what is",
            "define",
            "definition",
            "meaning"
        ]
    ):
        expansions.extend([
            "definition",
            "meaning"
        ])

    if any(
        word in q
        for word in [
            "advantage",
            "advantages",
            "benefit",
            "benefits"
        ]
    ):
        expansions.extend([
            "advantages",
            "benefits"
        ])

    if any(
        word in q
        for word in [
            "disadvantage",
            "disadvantages",
            "limitation",
            "limitations",
            "drawback"
        ]
    ):
        expansions.extend([
            "disadvantages",
            "limitations",
            "drawbacks"
        ])

    if any(
        word in q
        for word in [
            "type",
            "types",
            "kind",
            "kinds"
        ]
    ):
        expansions.extend([
            "types",
            "categories",
            "classification"
        ])

    if any(
        word in q
        for word in [
            "how does",
            "how do",
            "working",
            "work",
            "works"
        ]
    ):
        expansions.extend([
            "working",
            "process",
            "steps"
        ])

    if expansions:

        expansions = list(
            dict.fromkeys(expansions)
        )

        return (
            query
            + " "
            + " ".join(expansions)
        )

    return query


# ============================================================
# SEMANTIC SCORE
# ============================================================

def semantic_score(distance):

    # FAISS distance:
    # lower = better

    return math.exp(
        -max(distance, 0.0)
    )


# ============================================================
# KEYWORD SCORE
# ============================================================

def keyword_score(query, document):

    query_tokens = meaningful_tokens(query)
    document_tokens = meaningful_tokens(document)

    if not query_tokens or not document_tokens:
        return 0.0

    query_set = set(query_tokens)
    document_set = set(document_tokens)

    overlap = query_set.intersection(
        document_set
    )

    overlap_score = (
        len(overlap)
        / len(query_set)
    )

    # Frequency
    document_counter = Counter(
        document_tokens
    )

    frequency_score = 0.0

    for token in query_set:

        if token in document_counter:

            frequency_score += min(
                document_counter[token],
                3
            ) / 3

    frequency_score /= len(query_set)

    return min(
        0.70 * overlap_score
        + 0.30 * frequency_score,
        1.0
    )


# ============================================================
# INTENT SCORE
# ============================================================

def intent_score(query, document):

    intent = detect_query_intent(query)

    doc_lower = document.lower()

    score = 0.0

    if intent == "list":

        # Documents explicitly containing the requested
        # answer type get a boost.

        list_terms = [
            "examples",
            "example",
            "applications",
            "application",
            "types",
            "type",
            "advantages",
            "benefits",
            "disadvantages",
            "limitations",
            "use cases",
        ]

        for term in list_terms:

            if term in doc_lower:
                score = 1.0
                break

    elif intent == "definition":

        definition_terms = [
            "is",
            "defined",
            "definition",
            "refers to",
            "means",
            "subset",
        ]

        matches = sum(
            term in doc_lower
            for term in definition_terms
        )

        score = min(
            matches / 2,
            1.0
        )

    elif intent == "explanation":

        explanation_terms = [
            "process",
            "working",
            "steps",
            "works",
            "first",
            "then",
            "finally",
        ]

        matches = sum(
            term in doc_lower
            for term in explanation_terms
        )

        score = min(
            matches / 3,
            1.0
        )

    return score


# ============================================================
# DIRECTNESS SCORE
# ============================================================

def directness_score(query, document):

    query_tokens = meaningful_tokens(query)

    if not query_tokens:
        return 0.0

    document_tokens = meaningful_tokens(
        document
    )

    if not document_tokens:
        return 0.0

    query_set = set(query_tokens)

    matches = sum(
        token in document_tokens
        for token in query_set
    )

    coverage = (
        matches
        / len(query_set)
    )

    # Shorter focused chunks get a small bonus.
    #
    # This helps prefer a concise answer chunk
    # over a huge chapter chunk containing the same
    # terms many times.

    word_count = len(document_tokens)

    if word_count <= 120:
        length_bonus = 1.0

    elif word_count <= 250:
        length_bonus = 0.7

    elif word_count <= 500:
        length_bonus = 0.4

    else:
        length_bonus = 0.2

    return min(
        0.75 * coverage
        + 0.25 * length_bonus,
        1.0
    )


# ============================================================
# HYBRID RERANKER
# ============================================================

def rerank_documents(query, candidates):

    reranked = []

    for doc, distance in candidates:

        semantic = semantic_score(
            distance
        )

        lexical = keyword_score(
            query,
            doc.page_content
        )

        intent = intent_score(
            query,
            doc.page_content
        )

        directness = directness_score(
            query,
            doc.page_content
        )

        final_score = (
            SEMANTIC_WEIGHT * semantic
            + KEYWORD_WEIGHT * lexical
            + INTENT_WEIGHT * intent
            + DIRECTNESS_WEIGHT * directness
        )

        reranked.append(
            {
                "doc": doc,
                "final_score": final_score,
                "distance": distance,
                "semantic": semantic,
                "keyword": lexical,
                "intent": intent,
                "directness": directness,
            }
        )

    reranked.sort(
        key=lambda item: item["final_score"],
        reverse=True
    )

    return reranked


# ============================================================
# REMOVE DUPLICATES
# ============================================================

def remove_duplicates(reranked):

    seen = set()
    results = []

    for item in reranked:

        doc = item["doc"]

        chunk_id = doc.metadata.get(
            "chunk_id"
        )

        if chunk_id is None:

            chunk_id = (
                doc.metadata.get(
                    "source_file",
                    ""
                )
                + "_"
                + str(
                    doc.metadata.get(
                        "page_number",
                        doc.metadata.get(
                            "page",
                            ""
                        )
                    )
                )
                + "_"
                + doc.page_content[:100]
            )

        if chunk_id in seen:
            continue

        seen.add(chunk_id)

        results.append(item)

    return results


# ============================================================
# SELECT FINAL CONTEXT
# ============================================================

def select_final_results(
    query,
    reranked
):

    intent = detect_query_intent(
        query
    )

    # Simple factual/list questions need
    # less context to prevent answer pollution.

    if intent in {
        "list",
        "definition"
    }:

        final_k = 2

    else:

        final_k = 4

    return reranked[:final_k]


# ============================================================
# MAIN RETRIEVAL PIPELINE
# ============================================================

def retrieve_documents(
    query,
    initial_k=INITIAL_K,
    final_k=None
):

    original_query = clean_query(
        query
    )

    expanded_query = expand_query(
        original_query
    )

    # --------------------------------------------------------
    # FAISS candidate retrieval
    # --------------------------------------------------------

    candidates = (
        vector_db
        .similarity_search_with_score(
            expanded_query,
            k=initial_k
        )
    )

    # --------------------------------------------------------
    # Hybrid reranking
    # --------------------------------------------------------

    reranked = rerank_documents(
        original_query,
        candidates
    )

    # --------------------------------------------------------
    # Remove duplicate chunks
    # --------------------------------------------------------

    unique_results = remove_duplicates(
        reranked
    )

    # --------------------------------------------------------
    # Select final context
    # --------------------------------------------------------

    selected = select_final_results(
        original_query,
        unique_results
    )

    # Optional manual override
    if final_k is not None:
        selected = selected[:final_k]

    # Return only the document + scores
    # required by chatbot.py
    return [
        (
            item["doc"],
            item["final_score"],
            item["distance"],
            item["semantic"],
            item["keyword"],
            item["intent"],
            item["directness"],
        )
        for item in selected
    ]