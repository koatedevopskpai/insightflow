# app.py

# import os
from dotenv import load_dotenv
from openai import OpenAI

# load .env file in the folder
load_dotenv()

# Initialise the OpenAI Client & Get the API Key
client = OpenAI()


# create the Dad joke method
def generate_dad_joke():
    print("Thinking of a joke....\n")

    # use the client to get the OpenAI model to interact
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a hilarious dad. You only respond with hilarious jokes.",  # noqa : E501
            },
            {"role": "user", "content": "Tell me a joke!"},
        ],
        temperature=0.0,  # 0 is robotic/predictable. 1 is highly creative
    )
    # Extract the text from the response. It will be in json format
    joke = response.choices[0].message.content
    print(f"Dad bot says: {joke}")


if __name__ == "__main__":
    generate_dad_joke()
