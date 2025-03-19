import random

def number_guessing_game():
    number_to_guess = random.randint(1, 100)
    attempts = 0
    max_attempts = 10

    print("Welcome to the Number Guessing Game!")
    print("I have selected a number between 1 and 100. Can you guess it?")

    while attempts < max_attempts:
        guess = int(input("Guess the number (between 1 and 100): "))
        attempts += 1

        if guess < number_to_guess:
            print("Too low! Try again.")
        elif guess > number_to_guess:
            print("Too high! Try again.")
        else:
            print(f"Congratulations! You guessed it in {attempts} attempts!")
            break
    else:
        print(f"Game over! Better luck next time! The number was {number_to_guess}.")

number_guessing_game()
