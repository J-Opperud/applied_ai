## overview 
A collection of programs designed to build applied AI capabilities and validate concepts for future development.




## course_search.py

pre-trained AI model into your Python program.[hard coded search content]



 Python program
       │
       ▼
SentenceTransformer
       │
       ▼
all-MiniLM-L6-v2
       │
       ├── turns course sentences → numbers
       │
       └── turns your query → numbers
                 │
                 ▼
          similarity comparison


- Your query becomes a vector.
    query_embedding = model.encode(query)

- We compare that query vector against every course sentence vector.
    similarities = util.cos_sim(query_embedding, course_embeddings)[0]

     similarity
    Sentence 1 → 0.6823
    Sentence 2 → 0.4512

- selects the highest scores.
    top_results = similarities.topk(k=result_count)

- turns those results into the human-readable output required by the assignment.
    for rank, (score, sentence) in enumerate(results, start=1):
        print(f"  {rank}. [{score:.4f}] {sentence}")

smallest possible version of a retrieval system.
Uses:
    searching a company's documentation

    searching PDFs

    finding relevant support articles

    retrieving context for an LLM

    RAG (Retrieval-Augmented Generation)

    recommendation systems

architecture:

represent → retrieve → rank → present.

_____________________________________________


## Threshold experiment

DOCUMENTS
   ↓
extract text
   ↓
TF-IDF vectors
   ↓
query vector
   ↓
cosine similarity
   ↓
rank all 15 documents
   ↓
apply 0.3 / 0.5 / 0.7
   ↓
display results + missed category matches


example:
A threshold of 0.7 can reject a genuinely useful result.

"How can I record a song?"

0.34  Music producers use digital audio workstations to record and edit tracks.


Threshold 0.5 → 0 results
Threshold 0.7 → 0 results

experiment is showing that 0.5 and 0.7 are probably too aggressive for this particular TF-IDF dataset.

The appropriate threshold depends on the similarity model, documents, queries, and what we consider an acceptable match.

Keyword similarity ≠ semantic similarity

TF-IDF is useful, lightweight, and interpretable, but it doesn't understand language the way an embedding model can.


helpful why is to ground your query in truth:

Relevant category:
(
    "How do I manage dependencies in a PHP project?",
    {
        "Composer manages dependencies for PHP projects.",
    },
)


At 0.7, did this experiment missed a document that we had designated as relevant?

Stronger than 0.7 seems too high.

                         ┌── Expected relevant documents
                         │
Results from TF-IDF ─────┤
                         │
                         └── Compare
                              ↓
                         Did we miss anything?

Poor search result:
       
- Bad threshold
- Bad representation
- Bad query
- insufficient documents
  _____________________________________________

## Chunk and Compare

This project uses a pre-trained AI embedding model in a Python program to search a real PDF manual and compare two different chunking strategies.

The document used was the Hobart Hand Wrap Station Operation and Care manual.

                    Your Python program
                           │
                           ▼
                  SentenceTransformer
                           │
                           ▼
                    all-MiniLM-L6-v2
                           │
             ┌─────────────┴─────────────┐
             ▼                           ▼
       PDF document                  Your query
             │                           │
             ▼                           ▼
       text chunks                    vector
             │                           │
             └─────────────┬─────────────┘
                           ▼
                  similarity comparison
                           │
                           ▼
                    top 2 results


The PDF is first converted into text.

text = extract_pdf_text(PDF_FILE)

The document is divided using two different strategies.

Fixed-size:

300 characters per chunk

50 character overlap

Paragraph/section-based:

chunks are created around logical sections of the manual

Each chunk is converted into a numerical vector using the pre-trained model.

embeddings = model.encode(chunks)

The user's query is also converted into a vector.

query_embedding = model.encode(query)

The query vector is compared against the vectors for the document chunks using similarity scores.

Fixed-size chunk      → 0.6550
Paragraph chunk       → 0.6271


The program selects the two highest-scoring chunks for each query.

The results are displayed with their similarity scores and the actual text from the manual.

Three queries were tested:

How should the hand wrap station be operated?

What safety precautions should be followed?

How should the hand wrap station be cleaned and maintained?

The results showed that, for these three queries, the fixed-size strategy produced the higher average similarity score:

Fixed-size:       0.4867
Paragraph-based:  0.4161


The experiment demonstrated that chunking strategy can have a significant effect on retrieval quality. Fixed-size chunks performed better overall for this particular document and these queries, although some of the paragraph-based results also returned highly relevant information.

One important lesson from the experiment was that a higher similarity score does not automatically mean the result is better. The actual retrieved text also needs to be examined to determine whether it answers the question.

This project represents a small retrieval system that follows the basic architecture:

represent → retrieve → rank → present

It can be expanded into larger systems for:

searching company documentation

searching PDFs and manuals

finding relevant support articles

retrieving context for an LLM

RAG (Retrieval-Augmented Generation)

recommendation systems

document search and knowledge bases
______________________________________________________

## knowledge_base

Course notes
    │
    ▼
15+ documents
    │
    ├── id
    ├── document text
    └── metadata
          ├── module
          └── topic
    │
    ▼
ChromaDB persistent collection
    │
    ▼
search(query, module=None)
    │
    ├── semantic search
    ├── optional metadata filter
    └── top 5 results

- Persistent ChromaDB:
    client = chromadb.PersistentClient(path="./chroma_db")


- makes the script safe to run repeatedly:
    collection = client.get_or_create_collection(
        name="my_knowledge"
    )

 - documents:
    "id"
    "text"
    "metadata"
    metadata contains:

    {
        "module": "...",
        "topic": "..."
    }

- Upsert execute the script five times, you don't get five copies of every document: 
    collection.upsert(...)  

- search function:

    collection.query(
        query_texts=["How can I protect an API from attacks?"],
        n_results=5,
        where={"module": "5"},
    )

                      Query
                        │
                        ▼
                ┌─────────────────┐
                │ Metadata filter │
                │   module = 5    │
                └────────┬────────┘
                        │
                        ▼
                ┌─────────────────┐
                │ Semantic search │
                └────────┬────────┘
                         │
                         ▼
                       Top


- robustness 
    if not query.strip():
        raise ValueError("Query cannot be empty.")


The applied-AI connection

A simple ChromaDB exercise, architecturally building the first half of a RAG system.

documents → embeddings → retrieval → metadata filtering.
____________________________________________________________________________

## search_tool

 Streamlit semantic search application that uses ChromaDB to search course documents by meaning rather than exact keywords.

Architecture

Course Documents
      │
      ▼
 docs/ folder
      │
      ▼
load_and_chunk()
      │
      ▼
   ChromaDB
      │
      ├── stores text
      ├── stores source metadata
      └── creates vector representations
              │
              ▼
         User Query
              │
              ▼
       Semantic Search
              │
              ▼
       Ranked Results
              │
              ▼
        Streamlit UI

- How it works

    Loads documents

    chunks = load_and_chunk("docs")

    Stores them in ChromaDB

    collection.upsert(...)

    Turns the user's question into a semantic search

    results = collection.query(
        query_texts=[query],
        n_results=n_results,
    )

    Optionally filters by source

    where={"source": {"$in": selected_sources}}

    Ranks results using distance

    Document A → 0.32
    Document B → 0.61
    Document C → 1.12

    Lower distance generally means greater similarity.

    Displays results

    Showing 5 of 23 total documents

    Python Basics — chunk 3
    🟢 High (dist: 0.32)

    Python uses indentation to define...
    📖 Show full text
    🔎 Similar to this

- Assignment features

    Source filtering

    Result count

    150-character previews

    Expandable full text

    Relevance indicators

    Similar-to-this search

    Unique source-file count



ingest → represent → retrieve → rank → present

This is the foundation of:

    Documentation search

    PDF search

    Knowledge bases

    LLM context retrieval

    Recommendation systems

The application is essentially a small retrieval system—the same retrieval layer that can later be connected to an LLM for RAG.
