import chromadb


# ============================================================================
# DATA
# ============================================================================

DOCUMENTS = [
    {
        "id": "doc_1",
        "text": (
            "Python functions are reusable blocks of code defined with the "
            "def keyword. Functions can accept parameters and return values."
        ),
        "topic": "python",
    },
    {
        "id": "doc_2",
        "text": (
            "Python exceptions can be handled with try and except blocks. "
            "The finally block can be used for cleanup code."
        ),
        "topic": "python",
    },
    {
        "id": "doc_3",
        "text": (
            "Python lists are ordered collections that can contain multiple "
            "values. Items can be accessed using their index."
        ),
        "topic": "python",
    },
    {
        "id": "doc_4",
        "text": (
            "Object-oriented programming organizes software around objects "
            "that contain data and behavior. Classes define object structure."
        ),
        "topic": "programming",
    },
    {
        "id": "doc_5",
        "text": (
            "Clean code uses meaningful names, small functions, and simple "
            "control flow. The goal is to make software easier to maintain."
        ),
        "topic": "programming",
    },
    {
        "id": "doc_6",
        "text": (
            "Unit tests verify small pieces of software independently. "
            "Good tests check expected behavior as well as important edge cases."
        ),
        "topic": "testing",
    },
    {
        "id": "doc_7",
        "text": (
            "Integration tests verify that multiple components work together "
            "correctly. They test interactions between parts of a system."
        ),
        "topic": "testing",
    },
    {
        "id": "doc_8",
        "text": (
            "A test fixture provides consistent setup data for tests. "
            "Fixtures help tests avoid duplicated setup code."
        ),
        "topic": "testing",
    },
    {
        "id": "doc_9",
        "text": (
            "Semantic search retrieves documents based on meaning rather "
            "than requiring exact keyword matches."
        ),
        "topic": "ai",
    },
    {
        "id": "doc_10",
        "text": (
            "Embeddings represent text as numerical vectors. Similar meanings "
            "are represented by vectors that are close together."
        ),
        "topic": "ai",
    },
    {
        "id": "doc_11",
        "text": (
            "Retrieval evaluation commonly uses precision and recall. "
            "Precision measures the quality of returned results, while recall "
            "measures how many relevant results were found."
        ),
        "topic": "ai",
    },
    {
        "id": "doc_12",
        "text": (
            "A vector database stores embeddings and supports similarity "
            "search over high-dimensional numerical representations."
        ),
        "topic": "ai",
    },
]


EVAL_SET = [
    {
        "query": "How do I handle errors in Python?",
        "relevant_ids": ["doc_2"],
    },
    {
        "query": "What are reusable blocks of Python code?",
        "relevant_ids": ["doc_1"],
    },
    {
        "query": "How should I write maintainable software?",
        "relevant_ids": ["doc_5", "doc_4"],
    },
    {
        "query": "How do developers test software?",
        "relevant_ids": ["doc_6", "doc_7", "doc_8"],
    },
    {
        "query": "How does semantic search find similar meaning?",
        "relevant_ids": ["doc_9", "doc_10"],
    },
    {
        "query": "How do I measure search quality?",
        "relevant_ids": ["doc_11"],
    },
]


# ============================================================================
# CHROMADB SETUP
# ============================================================================

def create_collection():
    """Create and populate the ChromaDB collection."""
    client = chromadb.Client()

    collection = client.get_or_create_collection(
        name="evaluation_documents"
        )

    collection.add(
        ids=[doc["id"] for doc in DOCUMENTS],
        documents=[doc["text"] for doc in DOCUMENTS],
        metadatas=[
            {"topic": doc["topic"]}
            for doc in DOCUMENTS
            ],
        )

    return collection


# ============================================================================
# METRICS
# ============================================================================

def calculate_metrics(
    returned_ids: list[str],
    relevant_ids: list[str],
    ) -> tuple[float, float]:
    """
    
    Calculate precision and recall for one query.

    Precision:
        Of the documents returned, how many were relevant?

    Recall:
        Of the relevant documents, how many were returned?
    
    """
    returned = set(returned_ids)
    relevant = set(relevant_ids)

    true_positives = len(returned & relevant)

    precision = (
        true_positives / len(returned)
        if returned
        else 0.0
        )

    recall = (
        true_positives / len(relevant)
        if relevant
        else 0.0
        )

    return precision, recall



# EVALUATION
# ============================================================================

def evaluate(
    collection,
    n_results: int,
    threshold: float | None = None,
    show_distances: bool = False,
    ) -> tuple[float, float]:
    """
    Evaluate the search system.

    If threshold is None:
        Use all n_results returned by Chroma.

    If threshold is provided:
        Only keep documents whose distance is <= threshold.
    """

    precisions = []
    recalls = []

    for index, test_case in enumerate(EVAL_SET, start=1):

        results = collection.query(
            query_texts=[test_case["query"]],
            n_results=n_results,
            )

        candidate_ids = results["ids"][0]
        distances = results["distances"][0]

       
        # Optional debugging output
        # ------------------------------------------------------------

        if show_distances:
            print("  Candidates:")

            for doc_id, distance in zip(
                candidate_ids,
                distances,
            ):
                print(
                    f"    {doc_id}: "
                    f"distance={distance:.4f}"
                    )

        # ------------------------------------------------------------
        # Apply optional distance threshold
        # ------------------------------------------------------------

        if threshold is None:
            returned_ids = candidate_ids

        else:
            returned_ids = [
                doc_id
                for doc_id, distance in zip(
                    candidate_ids,
                    distances,
                )
                if distance <= threshold
                ]

            
        # Calculate metrics
        # ------------------------------------------------------------

        precision, recall = calculate_metrics(
            returned_ids,
            test_case["relevant_ids"],
            )

        precisions.append(precision)
        recalls.append(recall)

        print(
            f"Query {index}: "
            f"P={precision * 100:.1f}% "
            f"R={recall * 100:.1f}%"
            )

        print(f"  Query: {test_case['query']}")
        print(f"  Expected: {test_case['relevant_ids']}")
        print(f"  Returned: {returned_ids}")

    
    # Overall averages
    # ------------------------------------------------------------

    average_precision = sum(precisions) / len(precisions)
    average_recall = sum(recalls) / len(recalls)

    print(
        f"AVERAGE: "
        f"P={average_precision * 100:.1f}% "
        f"R={average_recall * 100:.1f}%"
        )

    return average_precision, average_recall


# EVALUATION RUNNERS
# ============================================================================

def run_n_results_evaluation(collection):
    """Evaluate using n_results without a distance threshold."""

    settings = [2, 3, 5]

    for n_results in settings:
        print()
        print("=" * 60)
        print(
            f"=== Evaluation with n_results={n_results} ===")
       
        print("=" * 60)

        evaluate(
            collection,
            n_results=n_results,
            show_distances=False,
            )


def run_threshold_evaluation(collection):
    """Evaluate using both threshold and n_results."""

    settings = [
        {"threshold": 0.7, "n_results": 3},
        {"threshold": 1.0, "n_results": 5},
        {"threshold": 1.5, "n_results": 5},
        ]

    for setting in settings:
        threshold = setting["threshold"]
        n_results = setting["n_results"]

        print()
        print("=" * 60)
        print(
            f"=== Evaluation at "
            f"threshold={threshold}, "
            f"n_results={n_results} ==="
            )
        print("=" * 60)

        evaluate(
            collection,
            n_results=n_results,
            threshold=threshold,
            show_distances=True,
            )


# ============================================================================
# MAIN
# ============================================================================

def main():
    collection = create_collection()

   
    # MODE 1: n_results only
    # ------------------------------------------------------------

    print()
    print("#" * 60)
    print("# MODE 1: n_results only")
    print("#" * 60)

    run_n_results_evaluation(collection)

    
    # MODE 2: threshold + n_results
    # ------------------------------------------------------------

    print()
    print("#" * 60)
    print("# MODE 2: threshold + n_results")
    print("#" * 60)

    run_threshold_evaluation(collection)

   
    # Analysis
    # ------------------------------------------------------------

    print()
    print("#" * 60)
    print("# ANALYSIS")
    print("#" * 60)

    print(
        """
The n_results experiments show how changing the number of retrieved
documents affects precision and recall.

A smaller n_results value limits the number of documents returned.
This can reduce irrelevant results, but it can also reduce recall.

The threshold experiments add another control. ChromaDB returns
candidate documents ranked by distance, and documents with a distance
greater than the threshold are removed.

A lower threshold is stricter because only very similar documents
are accepted.

A higher threshold is more permissive. This can improve recall by
allowing more relevant documents through, but it can also reduce
precision because irrelevant documents may also pass the filter.

The distance output is useful for diagnosing failures. For example,
some relevant documents have distances above 1.0, which means that
a threshold such as 0.3 is too strict for this dataset.

Potential improvements:
- Add more documents to each topic.
- Add more evaluation queries.
- Improve the quality of relevance judgments.
- Experiment with different embedding models.
- Tune the distance threshold using a larger evaluation set.
- Rewrite ambiguous queries or documents when appropriate.
"""
    )


if __name__ == "__main__":
    main()
