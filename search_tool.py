import os

import chromadb
import streamlit as st


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Semantic Search",
    page_icon="🔍",
    layout="wide",
    )


# ============================================================
# ChromaDB Setup
# ============================================================

@st.cache_resource
def get_collection():
    """Create the ChromaDB client and return the course collection."""
    client = chromadb.PersistentClient(path="./search_db")

    return client.get_or_create_collection(
        name="course_docs"
        )


collection = get_collection()


# ============================================================
# Document Loading & Chunking
# ============================================================

def load_and_chunk(directory):
    """Load .txt and .md files and split them into paragraphs."""
    chunks = []

    if not os.path.exists(directory):
        return chunks

    for filename in sorted(os.listdir(directory)):

        if not filename.endswith((".txt", ".md")):
            continue

        filepath = os.path.join(directory, filename)

        with open(filepath, "r", encoding="utf-8") as file:
            content = file.read()

        paragraphs = [
            paragraph.strip()
            for paragraph in content.split("\n\n")
            if paragraph.strip()
            ]

        for index, paragraph in enumerate(paragraphs):
            chunks.append(
                {
                    "text": paragraph,
                    "source": filename,
                    "chunk_id": f"{filename}_{index}",
                    "chunk_index": index,
                }
            )

    return chunks


# ============================================================
# Database Functions
# ============================================================

def index_documents():
    """Load documents from docs/ and add them to ChromaDB."""
    chunks = load_and_chunk("docs")

    if not chunks:
        return 0, 0

    collection.upsert(
        documents=[
            chunk["text"]
            for chunk in chunks
            ],
        metadatas=[
            {
                "source": chunk["source"],
                "chunk_index": str(chunk["chunk_index"]),
            }
            for chunk in chunks
        ],
        ids=[
            chunk["chunk_id"]
            for chunk in chunks
        ],
    )

    unique_sources = len(
        {
            chunk["source"]
            for chunk in chunks
        }
    )

    return len(chunks), unique_sources


def get_sources():
    """Return unique source filenames currently in ChromaDB."""
    if collection.count() == 0:
        return []

    results = collection.get(
        include=["metadatas"]
        )

    return sorted(
        {
            metadata["source"]
            for metadata in results["metadatas"]
            if metadata and "source" in metadata
        }
    )


def get_total_documents(selected_sources):
    """Return the number of chunks available for the current filter."""
    if selected_sources:
        results = collection.get(
            where={
                "source": {
                    "$in": selected_sources
                }
            },
            include=[],
            )
    else:
        results = collection.get(
            include=[]
            )

    return len(results["ids"])


# ============================================================
# Search Functions
# ============================================================

def search_documents(query, n_results, selected_sources=None):
    """Search ChromaDB and optionally filter by source file."""

    query_kwargs = {
        "query_texts": [query],
        "n_results": min(
            n_results,
            collection.count(),
        ),
    }

    if selected_sources:
        query_kwargs["where"] = {
            "source": {
                "$in": selected_sources
                }
            }

    return collection.query(**query_kwargs)


def get_relevance(distance):
    """Return a colored relevance label based on distance."""

    if distance < 0.5:
        return "🟢 High"

    if distance < 1.0:
        return "🟡 Medium"

    return "🔴 Low"


# ============================================================
# Result Display
# ============================================================

def display_results(results, show_similar_button=True):
    """Display search results in a consistent format."""

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    for index, (doc, metadata, distance) in enumerate(
        zip(
            documents,
            metadatas,
            distances,
        )
    ):

        relevance = get_relevance(distance)

        preview = doc[:150]

        if len(doc) > 150:
            preview += "..."

        with st.container():

            # ------------------------------------------------
            # Result Metadata
            # ------------------------------------------------

            col_meta, col_score = st.columns([3, 1])

            with col_meta:
                st.write(
                    f"**{metadata['source']}** "
                    f"— chunk {metadata['chunk_index']}"
                    )

            with col_score:
                st.write(
                    f"{relevance} "
                    f"(dist: {distance:.3f})"
                    )

            # ------------------------------------------------
            # Preview
            # ------------------------------------------------

            st.write(preview)

            # ------------------------------------------------
            # Full Text
            # ------------------------------------------------

            with st.expander("📖 Show full text"):
                st.write(doc)

            # ------------------------------------------------
            # Bonus: Similar Search
            # ------------------------------------------------

            if show_similar_button:

                if st.button(
                    "🔎 Similar to this",
                    key=f"similar_{index}",
                    ):
                    st.session_state["similar_query"] = doc
                    st.rerun()

            st.divider()


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.title("📁 Document Manager")

    # --------------------------------------------------------
    # Index Documents
    # --------------------------------------------------------

    if st.button(
        "🔄 Re-index Documents",
        use_container_width=True,
        ):

        chunk_count, source_count = index_documents()

        if chunk_count > 0:
            st.success(
                f"Indexed {chunk_count} chunks "
                f"from {source_count} files."
                )
        else:
            st.warning(
                "No .txt or .md files found in docs/."
                )

    # --------------------------------------------------------
    # Database Information
    # --------------------------------------------------------

    st.metric(
        "Documents in DB",
        collection.count(),
        )

    # --------------------------------------------------------
    # Source Filter
    # --------------------------------------------------------

    st.divider()

    sources = get_sources()

    selected_sources = st.multiselect(
        "Search only these files",
        options=sources,
        placeholder="All files",
        )

    # --------------------------------------------------------
    # Bonus: Unique Source Count
    # --------------------------------------------------------

    st.metric(
        "Unique source files",
        len(sources),
        )

    # --------------------------------------------------------
    # Result Count
    # --------------------------------------------------------

    n_results = st.slider(
        "Results to show",
        min_value=1,
        max_value=10,
        value=5,
        )


# ============================================================
# Main Search Interface
# ============================================================

st.title("🔍 Semantic Search")

st.write(
    "Search your course documents by meaning, "
    "not just keywords."
    )


# ============================================================
# Search Bar
# ============================================================

col_search, col_button = st.columns([5, 1])

with col_search:

    query = st.text_input(
        "Search your documents",
        placeholder="How does authentication work?",
        label_visibility="collapsed",
        )

with col_button:

    search_clicked = st.button(
        "🔍 Search",
        use_container_width=True,
        type="primary",
        )


# ============================================================
# Standard Search
# ============================================================

if search_clicked:

    if not query.strip():

        st.warning(
            "Please enter a search question."
            )

        st.stop()

    if collection.count() == 0:

        st.info(
            "No documents are indexed yet. "
            "Click 'Re-index Documents' in the sidebar."
            )

        st.stop()

    # Run the search
    results = search_documents(
        query=query,
        n_results=n_results,
        selected_sources=selected_sources,
        )

    documents = results["documents"][0]

    # Count documents available under current filter
    total_documents = get_total_documents(
        selected_sources
        )

    # --------------------------------------------------------
    # Result Count Display
    # --------------------------------------------------------

    st.subheader(
        f"Showing {len(documents)} "
        f"of {total_documents} total documents"
        )

    # --------------------------------------------------------
    # Display Results
    # --------------------------------------------------------

    display_results(results)


# ============================================================
# Similar Search
# ============================================================

if "similar_query" in st.session_state:

    similar_query = st.session_state["similar_query"]

    st.divider()

    st.subheader(
        "🔎 Similar Documents"
        )

    st.caption(
        "Searching for documents similar "
        "to the selected result."
        )

    if collection.count() > 0:

        similar_results = search_documents(
            query=similar_query,
            n_results=n_results,
            selected_sources=selected_sources,
            )

        similar_documents = (
            similar_results["documents"][0]
            )

        st.write(
            f"Showing {len(similar_documents)} "
            f"similar results"
            )

        display_results(
            similar_results,
            show_similar_button=False,
            )


# ============================================================
# Empty Database Message
# ============================================================

if (
    collection.count() == 0
    and not search_clicked
    ):

    st.info(
        "👈 Click 'Re-index Documents' in the sidebar "
        "to load your documents first."
        )
