

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DOCUMENTS = [
    # PHP development
    ("php", "Laravel provides tools for building modern PHP web applications."),
    ("php", "Composer manages dependencies for PHP projects."),
    ("php", "PHP applications can connect to MySQL databases using PDO."),
    ("php", "PHP developers use PHPUnit to write automated tests."),

    # Books
    ("books", "Science fiction novels often explore technology and future societies."),
    ("books", "A mystery novel usually revolves around solving a crime or puzzle."),
    ("books", "Book clubs discuss themes, characters, and ideas from selected books."),
    ("books", "Biographies tell the life story of a real person."),

    # Music
    ("music", "Jazz musicians frequently improvise over chord progressions."),
    ("music", "A guitar can be used to play melodies, chords, and rhythmic patterns."),
    ("music", "Music producers use digital audio workstations to record and edit tracks."),
    ("music", "Classical composers use instruments to create complex musical arrangements."),

    # Weather
    ("weather", "Meteorologists use radar to track precipitation and severe storms."),
    ("weather", "A cold front can cause temperatures to drop and storms to develop."),
    ("weather", "Weather forecasts use atmospheric data to predict future conditions."),
]


QUERIES = [
    (
        "How do I manage dependencies in a PHP project?",
        {
            "Composer manages dependencies for PHP projects.",
        },
    ),
    (
        "What kind of book should I read?",
        {
            "Science fiction novels often explore technology and future societies.",
            "A mystery novel usually revolves around solving a crime or puzzle.",
            "Biographies tell the life story of a real person.",
        },
    ),
    (
        "How can I record a song?",
        {
            "Music producers use digital audio workstations to record and edit tracks.",
        },
    ),
    (
        "Will there be storms tomorrow?",
        {
            "Meteorologists use radar to track precipitation and severe storms.",
            "A cold front can cause temperatures to drop and storms to develop.",
            "Weather forecasts use atmospheric data to predict future conditions.",
        },
    ),
    (
        "How can I prepare for changing conditions?",
        {
            "Weather forecasts use atmospheric data to predict future conditions.",
        },
    ),
]



THRESHOLDS = (0.3, 0.5, 0.7)


def calculate_scores(
    query: str,
    documents: list[tuple[str, str]],
    vectorizer: TfidfVectorizer,
) -> list[tuple[float, str, str]]:
    """Return documents ranked by similarity to the query."""

    document_texts = [text for _, text in documents]

    document_vectors = vectorizer.transform(document_texts)
    query_vector = vectorizer.transform([query])

    similarities = cosine_similarity(
        query_vector,
        document_vectors,
    )[0]

    results = [
        (float(score), category, document)
        for score, (category, document)
        in zip(similarities, documents)
    ]

    return sorted(results, reverse=True)


def display_results(
    query: str,
    relevant_category: str,
    scored_documents: list[tuple[float, str, str]],
    thresholds: tuple[float, ...],
) -> None:
    """Display results for each similarity threshold."""

    print(f'\nQuery: "{query}"')
    print(f"Relevant category: {relevant_category}")

    for threshold in thresholds:
        passing_results = [
            (score, category, document)
            for score, category, document in scored_documents
            if score >= threshold
        ]

        missed_relevant_results = [
            (score, category, document)
            for score, category, document in scored_documents
            if category == relevant_category and score < threshold
        ]

        print(
            f"\n  Threshold {threshold:.1f}: "
            f"{len(passing_results)} results"
        )

        if passing_results:
            for score, category, document in passing_results:
                print(f"    [{score:.2f}] {document}")
        else:
            print("    No results passed this threshold.")

        if missed_relevant_results:
            print("    Missed relevant results:")

            for score, _, document in missed_relevant_results:
                print(f"      [{score:.2f}] {document}")


def main() -> None:
    """Run the similarity threshold experiment."""

    document_texts = [text for _, text in DOCUMENTS]

    vectorizer = TfidfVectorizer(stop_words="english")
    vectorizer.fit(document_texts)

    for query, relevant_category in QUERIES:
        scored_documents = calculate_scores(
            query,
            DOCUMENTS,
            vectorizer,
            )

        display_results(
            query,
            relevant_category,
            scored_documents,
            THRESHOLDS,
            )


if __name__ == "__main__":
    main()
