## overview 
A collection of programs designed to build applied AI capabilities and validate concepts for future development.




## course_search.py

pre-trained AI model into your Python program.[hard coded search content]

Think of it like:

Your Python program
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
