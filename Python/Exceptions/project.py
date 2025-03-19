import logging

# Configure logging
logging.basicConfig(filename='error_log.txt', level=logging.ERROR, format='%(asctime)s:%(levelname)s:%(message)s')

def get_number(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Invalid input! Please enter a valid number.")
            logging.error("ValueError occurred: invalid input.")

def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

def divide(a, b):
    try:
        return a / b
    except ZeroDivisionError:
        print("Oops! Division by zero is not allowed.")
        logging.error("ZeroDivisionError occurred: division by zero.")
        return None

def menu():
    while True:
        print("\nWelcome to the Error-Free Calculator!")
        print("Choose an operation:")
        print("1. Addition")
        print("2. Subtraction")
        print("3. Multiplication")
        print("4. Division")
        print("5. Exit")

        choice = input("> ")

        if choice == '5':
            print("Goodbye!")
            break

        if choice in ['1', '2', '3', '4']:
            num1 = get_number("Enter the first number: ")
            num2 = get_number("Enter the second number: ")

            if choice == '1':
                result = add(num1, num2)
            elif choice == '2':
                result = subtract(num1, num2)
            elif choice == '3':
                result = multiply(num1, num2)
            elif choice == '4':
                result = divide(num1, num2)

            if result is not None:
                print(f"The result is {result}.")
        else:
            print("Invalid choice. Please select a valid option.")

menu()
