from transformers import pipeline


generator = pipeline("text-generation", model="distilgpt2")

prompt_chain_of_thought = """
Step 1: Think about the theme of the poem.
Step 2: Choose a poetic structure, such as a sonnet or free verse.
Step 3: Write the first line of the poem.
Step 4: Continue writing the poem, ensuring each line flows naturally from the last.
Step 5: Conclude the poem with a strong finishing line.

Theme: Nature's beauty
"""

prompt_few_shot = """
Here are some examples of poetry:

Example 1:
The road not taken by Robert Frost
Two roads diverged in a yellow wood,
And sorry I could not travel both
And be one traveler, long I stood
And looked down one as far as I could

Example 2:
She walks in beauty by Lord Byron
She walks in beauty, like the night
Of cloudless climes and starry skies;
And all that's best of dark and bright
Meet in her aspect and her eyes:

Now, write a poem on the theme of 'Nature's beauty'.
"""

prompt_role_play = """
Act as a renowned poet from the Romantic era. Compose a poem that captures the essence of 'Nature's beauty'. Ensure the language is rich and evocative, reflecting the style of the era.
"""

response_chain_of_thought = generator(prompt_chain_of_thought, max_new_tokens=50, num_return_sequences=1)
response_few_shot = generator(prompt_few_shot, max_new_tokens=50, num_return_sequences=1)
response_role_play = generator(prompt_role_play, max_new_tokens=50, num_return_sequences=1)

print("Chain of Thought Prompting:\n", response_chain_of_thought[0]['generated_text'])
print("\nFew-Shot Learning:\n", response_few_shot[0]['generated_text'])
print("\nRole Play:\n", response_role_play[0]['generated_text'])
