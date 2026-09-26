from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import numpy as np


PDF_FILE = "617-625A1_manual.pdf"








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
#cleaning text to remove large white spaces like pictures and graphs 
# ---------------------------------------------------------

def clean_text(text):
    lines = text.splitlines()
    cleaned_lines = []

    for line in lines:
        line = line.strip()

        if line:
            cleaned_lines.append(line)

    # Turn the remaining lines into normal paragraphs
    return "\n\n".join(cleaned_lines)

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
    lines = text.splitlines()

    chunks = []
    current_chunk = ""

    section_headings = {
        "GENERAL",
        "W3-2 AND W3-2C FILM LOADING AND THREADING",
        "CHANGING FILM WIDTHS",
        "625A FILM LOADING AND THREADING",
        "WRAPPING PACKAGES",
        "CLEANING",
        "MAINTENANCE"
        }

    for line in lines:
        line = line.strip()

        if not line:
            continue

        # Normalize the line for checking headings
        heading_check = line.upper()

        # Start a new chunk when we reach a section heading
        if heading_check in section_headings:

            if current_chunk:
                chunks.append(current_chunk.strip())

            current_chunk = line

        else:
            if current_chunk:
                current_chunk += " " + line
            else:
                current_chunk = line

    # Add final chunk
    if current_chunk:
        chunks.append(current_chunk.strip())

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

    print("\n" + "=" * 70)
    print("RAW CLEANED TEXT")
    print("=" * 70)

    print(text[:5000])

    print("=" * 70)
    input("Press Enter to continue...")


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
Written Analysis:

In this experiment, the fixed-size chunking strategy produced the
higher average similarity score. The fixed-size chunks had an average
top-2 similarity score of 0.4867, compared with 0.4161 for the
paragraph-based chunks.

The results were fairly close for the operation query, with scores of
0.5405 for fixed-size chunking and 0.5373 for paragraph-based
chunking. For the safety query, fixed-size chunking produced a higher
score of 0.4018 compared with 0.3131. For the cleaning and maintenance
query, fixed-size chunking also produced a higher score of 0.6550
compared with 0.6271.

One reason fixed-size chunking performed well is that the 300-character
chunks can focus on a specific part of an instruction while still
using 50 characters of overlap to preserve some context between
chunks. Paragraph-based chunks preserve larger sections of the
document, but larger chunks can contain additional information that
is not directly related to the query.

The results demonstrate that there is no single chunking strategy that
is always best. The best approach depends on the structure of the
document and the type of information being searched. For this
particular manual and these three queries, fixed-size chunking produced
the stronger retrieval scores.
""")


    print("=" * 70)


if __name__ == "__main__":
    main()
