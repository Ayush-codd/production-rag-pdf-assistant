from search import retrieve_documents
from chatbot import generate_answer_with_sources


def main():

    print("\n🤖 RAG Assistant")
    print("Ask questions about your documents.")
    print("Type 'exit' to quit.\n")

    while True:

        try:

            query = input("You: ").strip()

        except (
            KeyboardInterrupt,
            EOFError
        ):

            print("\nGoodbye!")
            break

        if not query:
            continue

        if query.lower() in {
            "exit",
            "quit",
            "bye"
        }:

            print("Goodbye!")
            break

        try:

            results = retrieve_documents(
                query
            )

            answer = generate_answer_with_sources(
                query,
                results
            )

            print(
                f"\nAssistant: {answer}\n"
            )

        except Exception as e:

            print(
                "\nAssistant: "
                "I couldn't process that question right now.\n"
            )


if __name__ == "__main__":
    main()