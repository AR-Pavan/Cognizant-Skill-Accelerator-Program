import openai
import os
from dotenv import load_dotenv

load_dotenv()

# Set your OpenAI API key here
openai.api_key = os.getenv("MY_API_KEY")

if openai.api_key:
    print("API Key found:")
else:
    print("API Key not found.")

def chat_with_gpt(prompt, model="gpt-3.5-turbo", max_tokens=100, temperature=0.7):
    """
    Function to generate text using OpenAI's GPT model.
    :param prompt: User's input text
    :param model: AI model to use (default: "gpt-3.5-turbo")
    :param max_tokens: Max length of response
    :param temperature: Creativity level (0.0 = strict, 1.0 = very creative)
    :return: Generated text
    """
    try:
        client = openai.OpenAI()  # Create OpenAI client instance
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    print("Welcome to AI-Powered Text Completion!")
    while True:
        user_prompt = input("\nEnter your prompt (or type 'exit' to quit): ")
        if user_prompt.lower() == "exit":
            print("Goodbye!")
            break
        response = chat_with_gpt(user_prompt)
        print("\nAI Response:\n", response)
