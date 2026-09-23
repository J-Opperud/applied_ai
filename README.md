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