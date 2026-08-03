import uuid
import requests
import questionary


BASE_URL = "http://localhost:5000"


def ask_question(question):

    try:
        response = requests.post(
            f"{BASE_URL}/question",
            json={"question": question},
            timeout=120,
        )

        response.raise_for_status()
        return response.json()

    except requests.RequestException as e:
        print(f"\nRequest failed: {e}")
        return None


def send_feedback(conversation_id, feedback):

    response = requests.post(
        f"{BASE_URL}/feedback",
        json={
            "conversation_id": conversation_id,
            "feedback": feedback,
        },
    )

    return response.status_code


def main():

    print("===================================")
    print(" Harvard CS249R Study Assistant")
    print("===================================")

    while True:

        question = questionary.text(
            "Ask your question:"
        ).ask()

        if not question:
            break


        result = ask_question(question)

        if not result:
            continue


        print("\nAnswer:\n")
        print(result.get("answer"))


        sources = result.get("sources", [])

        if sources:
            print("\nSources:")
            for source in sources:
                print(f"- {source}")


        conversation_id = result.get(
            "conversation_id",
            str(uuid.uuid4())
        )


        feedback = questionary.select(
            "Rate this answer:",
            choices=[
                "👍 Good",
                "👎 Bad",
                "Skip",
            ],
        ).ask()


        if feedback != "Skip":

            value = 1 if feedback == "👍 Good" else -1

            status = send_feedback(
                conversation_id,
                value,
            )

            print(
                f"Feedback saved (HTTP {status})"
            )


        again = questionary.confirm(
            "Ask another question?"
        ).ask()

        if not again:
            break


    print("Goodbye!")


if __name__ == "__main__":
    main()