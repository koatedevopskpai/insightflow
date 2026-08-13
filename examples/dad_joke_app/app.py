# app.py
from dotenv import load_dotenv
from openai import OpenAI

# Load secrets and initialize client
load_dotenv()
client = OpenAI()


def chat_loop():
    print("🤖 Welcome to the Dad Joke Terminal! (Type 'exit' to quit)")

    # This loop keeps running forever until we explicitly break it
    while True:
        # 1. Wait for the user to type something
        user_topic = input("\nWhat topic do you want a joke about? > ")

        # 2. Give the user a way to escape the loop
        if user_topic.lower() == "exit":
            print("Catch you later!")
            break

        # 3. Call the API,
        # dynamically inserting the user's topic using an f-string
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a hilarious dad. You only respond with short, punchy dad jokes about the specific topic the user provides.",  # noqa : E501
                },
                {
                    "role": "user",
                    "content": f"Tell me a joke about {user_topic}",
                },  # <--- Dynamic injection!
            ],
            temperature=0.8,
        )

        # 4. Print the result
        joke = response.choices[0].message.content
        print(f"👨 Dad Bot: {joke}")


if __name__ == "__main__":
    chat_loop()
