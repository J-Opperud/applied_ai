import chromadb


# Persistent storage location.
DB_PATH = "./chroma_db"
COLLECTION_NAME = "my_knowledge"


DOCUMENTS = [
    # -------------------------
    # Module 4: Web / HTTP
    # -------------------------
    {
        "id": "m4-http-001",
        "text": (
            "DNS translates human-readable domain names such as google.com "
            "into IP addresses that computers use to locate servers."
        ),
        "metadata": {"module": "4", "topic": "web"},
    },
    {
        "id": "m4-http-002",
        "text": (
            "HTTP is the protocol browsers and servers use to communicate. "
            "Requests contain information such as the method, path, headers, and sometimes a body."
        ),
        "metadata": {"module": "4", "topic": "http"},
    },
    {
        "id": "m4-http-003",
        "text": (
            "HTTP status codes communicate the result of a request. "
            "For example, 200 means success, 301 means permanent redirection, "
            "404 means not found, and 500 indicates a server error."
        ),
        "metadata": {"module": "4", "topic": "http"},
    },
    {
        "id": "m4-http-004",
        "text": (
            "HTTPS encrypts communication between a browser and server, "
            "which is essential when applications handle sensitive information."
        ),
        "metadata": {"module": "4", "topic": "security"},
    },

    # -------------------------
    # Module 5: API / Security
    # -------------------------
    {
        "id": "m5-security-001",
        "text": (
            "CORS controls which origins are allowed to make requests to a FastAPI application "
            "from a browser."
        ),
        "metadata": {"module": "5", "topic": "api"},
    },
    {
        "id": "m5-security-002",
        "text": (
            "In production, CORS should allow specific trusted origins instead of allowing every origin."
        ),
        "metadata": {"module": "5", "topic": "security"},
    },
    {
        "id": "m5-security-003",
        "text": (
            "SQL injection occurs when untrusted input is inserted into SQL statements as executable code. "
            "SQLAlchemy's ORM helps prevent this by parameterizing values."
        ),
        "metadata": {"module": "5", "topic": "database"},
    },
    {
        "id": "m5-security-004",
        "text": (
            "When raw SQL is necessary, parameterized queries should be used instead of string formatting "
            "to keep user input separate from SQL code."
        ),
        "metadata": {"module": "5", "topic": "database"},
    },
    {
        "id": "m5-security-005",
        "text": (
            "Rate limiting restricts how frequently clients can call an API and helps prevent abuse. "
            "The slowapi library can provide rate limiting for FastAPI."
        ),
        "metadata": {"module": "5", "topic": "security"},
    },
    {
        "id": "m5-security-006",
        "text": (
            "Pydantic input validation is also a security practice because field lengths and accepted "
            "formats can prevent unexpectedly large or malformed payloads."
        ),
        "metadata": {"module": "5", "topic": "validation"},
    },

    # -------------------------
    # Module 6: Streamlit / Frontend
    # -------------------------
    {
        "id": "m6-streamlit-001",
        "text": (
            "Streamlit turns Python scripts into interactive web applications without requiring separate "
            "HTML, CSS, and JavaScript files."
        ),
        "metadata": {"module": "6", "topic": "frontend"},
    },
    {
        "id": "m6-streamlit-002",
        "text": (
            "Streamlit re-runs the entire Python script whenever a user interacts with a widget."
        ),
        "metadata": {"module": "6", "topic": "streamlit"},
    },
    {
        "id": "m6-streamlit-003",
        "text": (
            "Streamlit session state stores values that need to survive script re-runs, "
            "such as counters, chat history, and application state."
        ),
        "metadata": {"module": "6", "topic": "state"},
    },
    {
        "id": "m6-streamlit-004",
        "text": (
            "Streamlit forms batch widget inputs so that changing individual fields does not trigger "
            "a script re-run until the form is submitted."
        ),
        "metadata": {"module": "6", "topic": "streamlit"},
    },
    {
        "id": "m6-streamlit-005",
        "text": (
            "Streamlit's st.cache_data can cache expensive data-fetching functions so repeated "
            "script re-runs do not unnecessarily repeat the same work."
        ),
        "metadata": {"module": "6", "topic": "performance"},
    },

    # -------------------------
    # Module 7: Data / AI
    # -------------------------
    {
        "id": "m7-pandas-001",
        "text": (
            "A pandas DataFrame is a two-dimensional data structure with rows and columns, "
            "similar to a spreadsheet."
        ),
        "metadata": {"module": "7", "topic": "data"},
    },
    {
        "id": "m7-pandas-002",
        "text": (
            "Pandas filtering creates subsets of rows using boolean conditions such as salary greater "
            "than a threshold or department equal to Engineering."
        ),
        "metadata": {"module": "7", "topic": "pandas"},
    },
    {
        "id": "m7-pandas-003",
        "text": (
            "Pandas groupby and aggregation can summarize large datasets by calculating values such as "
            "average salary, total revenue, or headcount."
        ),
        "metadata": {"module": "7", "topic": "pandas"},
    },
    {
        "id": "m7-ai-001",
        "text": (
            "Understanding algorithmic complexity matters in AI engineering because inefficient operations "
            "can become extremely slow as datasets grow."
        ),
        "metadata": {"module": "7", "topic": "ai"},
    },
    {
        "id": "m7-ai-002",
        "text": (
            "A set provides average constant-time membership checks, making it useful for detecting "
            "duplicates in linear time."
        ),
        "metadata": {"module": "7", "topic": "algorithms"},
    },
]


def create_collection():
    """Create or retrieve the persistent knowledge collection."""
    client = chromadb.PersistentClient(path=DB_PATH)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    return collection


def load_documents(collection):
    """Upsert course documents into the collection."""
    collection.upsert(
        ids=[document["id"] for document in DOCUMENTS],
        documents=[document["text"] for document in DOCUMENTS],
        metadatas=[document["metadata"] for document in DOCUMENTS],
    )


def search(collection, query, module=None):
    """
    Search the knowledge base and return up to five relevant documents.

    Args:
        collection: ChromaDB collection to search.
        query: Natural-language search query.
        module: Optional module number used as a metadata filter.

    Returns:
        ChromaDB query results.
    """
    if not query.strip():
        raise ValueError("Query cannot be empty.")

    query_kwargs = {
        "query_texts": [query],
        "n_results": 5,
    }

    if module is not None:
        query_kwargs["where"] = {"module": str(module)}

    return collection.query(**query_kwargs)


def display_results(results):
    """Print search results in a readable format."""
    documents = results["documents"][0]
    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    ids = results["ids"][0]

    for rank, (doc_id, document, distance, metadata) in enumerate(
        zip(ids, documents, distances, metadatas),
        start=1,
    ):
        print(f"\n{rank}. {doc_id}")
        print(f"   Distance: {distance:.4f}")
        print(f"   Module:   {metadata['module']}")
        print(f"   Topic:    {metadata['topic']}")
        print(f"   Content:  {document}")


def run_demo(collection):
    """Run the assignment's required search demonstrations."""

    print("\n" + "=" * 60)
    print("TEST 1: Broad search")
    print("=" * 60)

    results = search(
        collection,
        "How do applications communicate over the internet?",
    )
    display_results(results)

    print("\n" + "=" * 60)
    print("TEST 2: Module-filtered search")
    print("=" * 60)

    results = search(
        collection,
        "How can I protect an API from attacks?",
        module="5",
    )
    display_results(results)

    print("\n" + "=" * 60)
    print("TEST 3: Different wording")
    print("=" * 60)

    results = search(
        collection,
        "How can software stop someone from abusing an endpoint?",
    )
    display_results(results)


def main():
    collection = create_collection()
    load_documents(collection)

    print(f"Collection: {collection.name}")
    print(f"Documents:  {collection.count()}")

    run_demo(collection)


if __name__ == "__main__":
    main()
