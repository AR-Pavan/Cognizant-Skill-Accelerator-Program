
# Ask the user to input a word
word = input("Enter a word: ")

# Reverse the string using slicing
reversed_word = word[::-1]

# Check if the original word is the same as the reversed word
if word == reversed_word:
    print(f"Yes, '{word}' is a palindrome!")
else:
    print(f"No, '{word}' is not a palindrome.")
