import openai
import os


openai.api_key = "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"

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
        client = openai.OpenAI()  
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
