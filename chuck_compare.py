from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np


PDF_FILE = "617-625A1_manual.pdf"






def clean_text(text):
    # Replace repeated whitespace with a single space
    lines = text.splitlines()

    cleaned_lines = []

    for line in lines:
        line = line.strip()

        if line:
            cleaned_lines.append(line)

    # Reconstruct paragraphs
    return "\n\n".join(cleaned_lines)

# ---------------------------------------------------------
# 1. Extract text from the PDF
# ---------------------------------------------------------

def extract_pdf_text(filename):
    reader = PdfReader(filename)

    pages = []

    for page in reader.pages:
        text = page.extract_text()

        if text:
            pages.append(text)

    return "\n\n".join(pages)


# ---------------------------------------------------------
# 2. Fixed-size chunking
#    300 characters with 50 character overlap
# ---------------------------------------------------------

def fixed_size_chunks(text, chunk_size=300, overlap=50):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        # Move forward while keeping the requested overlap
        start += chunk_size - overlap

    return chunks


# ---------------------------------------------------------
# 3. Paragraph-based chunking
#    Split on double newlines
# ---------------------------------------------------------

def paragraph_chunks(text):
    paragraphs = text.split("\n\n")

    chunks = []

    for paragraph in paragraphs:
        paragraph = " ".join(paragraph.split())

        if paragraph:
            chunks.append(paragraph)

    return chunks


# ---------------------------------------------------------
# 4. Embed chunks
# ---------------------------------------------------------

def embed_chunks(model, chunks):
    return model.encode(
        chunks,
        convert_to_numpy=True,
        normalize_embeddings=True
    )


# ---------------------------------------------------------
# 5. Search for the most relevant chunks
# ---------------------------------------------------------

def search(query, model, chunks, embeddings, top_k=2):
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True
    )[0]

    # Because embeddings are normalized, dot product = cosine similarity
    scores = np.dot(embeddings, query_embedding)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:
        results.append((chunks[index], scores[index]))

    return results


# ---------------------------------------------------------
# 6. Main program
# ---------------------------------------------------------

def main():

    print("=" * 70)
    print("CHUNKING STRATEGY COMPARISON")
    print("=" * 70)

    # Extract document
    print("\nReading PDF...")
    text = extract_pdf_text(PDF_FILE)
    text = clean_text(text)

    print(f"Extracted approximately {len(text):,} characters.")

    # Create both types of chunks
    fixed_chunks = fixed_size_chunks(text)
    paragraph_based_chunks = paragraph_chunks(text)

    print("\nChunk counts:")
    print(f"Fixed-size chunks:   {len(fixed_chunks)}")
    print(f"Paragraph chunks:    {len(paragraph_based_chunks)}")

    # Load embedding model
    print("\nLoading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Embed both strategies
    print("Embedding fixed-size chunks...")
    fixed_embeddings = embed_chunks(model, fixed_chunks)

    print("Embedding paragraph chunks...")
    paragraph_embeddings = embed_chunks(
        model,
        paragraph_based_chunks
    )

    # -----------------------------------------------------
    # Queries
    # -----------------------------------------------------

    queries = [
        "How should the hand wrap station be operated?",
        "What safety precautions should be followed?",
        "How should the hand wrap station be cleaned and maintained?"
    ]

    # Store scores so we can compare strategies later
    fixed_scores_all = []
    paragraph_scores_all = []

    # -----------------------------------------------------
    # Run searches
    # -----------------------------------------------------

    for query_number, query in enumerate(queries, start=1):

        print("\n" + "=" * 70)
        print(f"QUERY {query_number}: {query}")
        print("=" * 70)

        # Fixed-size results
        fixed_results = search(
            query,
            model,
            fixed_chunks,
            fixed_embeddings,
            top_k=2
        )

        print("\n--- FIXED-SIZE CHUNKS ---")

        for number, (chunk, score) in enumerate(fixed_results, start=1):

            print(f"\nResult {number}")
            print(f"Score: {score:.4f}")
            print(f"Text: {chunk}")

            fixed_scores_all.append(score)

        # Paragraph results
        paragraph_results = search(
            query,
            model,
            paragraph_based_chunks,
            paragraph_embeddings,
            top_k=2
        )

        print("\n--- PARAGRAPH-BASED CHUNKS ---")

        for number, (chunk, score) in enumerate(
            paragraph_results,
            start=1
        ):

            print(f"\nResult {number}")
            print(f"Score: {score:.4f}")
            print(f"Text: {chunk}")

            paragraph_scores_all.append(score)

    # -----------------------------------------------------
    # Final comparison
    # -----------------------------------------------------

    fixed_average = np.mean(fixed_scores_all)
    paragraph_average = np.mean(paragraph_scores_all)

    print("\n" + "=" * 70)
    print("FINAL COMPARISON")
    print("=" * 70)

    print(f"\nAverage top-2 similarity score:")
    print(f"Fixed-size:       {fixed_average:.4f}")
    print(f"Paragraph-based:  {paragraph_average:.4f}")

    print("\nWritten Analysis:")
    print("""
For this document, paragraph-based chunking is expected to provide
better search quality because the Hobart hand wrap station manual is
organized into logical sections and instructions. Paragraph-based
chunks preserve complete ideas and keep related instructions together.

The fixed-size strategy is simple and predictable, but it can split a
sentence or procedure in the middle. This can cause a retrieved chunk
to contain only part of the information needed to answer a query.
Fixed-size chunks can also include pieces of two unrelated topics.

Paragraph-based chunking better preserves the natural structure of
the manual. For example, information about operation, safety, cleaning,
and maintenance can remain together instead of being divided simply
because a character limit was reached.

The fixed-size approach can still be useful when documents contain
very long paragraphs or when consistent chunk sizes are important.
However, for this relatively short instruction manual, preserving
paragraph boundaries provides more meaningful context for semantic
search.

Therefore, for this particular document, paragraph-based chunking is
the strategy I would choose. The search results and similarity scores
above provide the practical comparison between the two approaches.
""")

    print("=" * 70)


if __name__ == "__main__":
    main()
