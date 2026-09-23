from sentence_transformers import SentenceTransformer, util

course_content =[
    "FastAPI automatically validates request data using Pydantic models.","Pydantic schemas define what valid data looks like.",
    "Error handling in FastAPI returns clear messages for invalid requests.",
    "Python functions help organize code into reusable pieces.",
    "List comprehensions provide a concise way to transform collections.",
    "Dictionaries store data as key-value pairs and are useful for structured information.",
    "Environment variables can keep configuration and secrets outside of source code.",
    "APIs allow different software systems to communicate with each other.",
    "HTTP methods such as GET and POST describe different kinds of requests.",
    "JSON is commonly used to exchange structured data between applications.",
    "Virtual environments isolate a project's Python dependencies.",
    "Semantic search finds relevant information based on meaning rather than exact keywords.",
    ]

def search_course(
        query: str,
        model: SentenceTransformer,
        course_content:  list[str],
        course_embeddings,top_k: int = 3,
        ) -> list[tuple[float, str]]:
    """Find the most semantically similar course sentences."""

    query_embedding = model.encode(query)

    similarities = util.cos_sim(query_embedding, course_embeddings)[0]

    result_count = min(
        top_k, len(course_content)
        )
    top_results = similarities.topk(
        k=result_count
        )

    results = [
        (score.item(), course_content[index])
        for score, 
        index in zip(
            top_results.values, 
            top_results.indices
            )
        ]

    return results


def main() -> None:
    """Run the interactive course search application."""

    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Loading course content...")
    course_embeddings = model.encode(course_content)

    while True:
        query = input("\nSearch (or 'quit): ").strip()

        if query.lower() == "quit":
            print("Goodbye!")
            break

        if not query:
            print("Please enter a search query.")
            continue

        results = search_course(
            query,
            model,
            course_content,
            course_embeddings,
        )

        print("\nTop 3 results:")

        for rank, (score, sentence) in enumerate(results, start=1):
            print(f"  {rank}. [{score:.4f}] {sentence}")


if __name__ == "__main__":
    main()