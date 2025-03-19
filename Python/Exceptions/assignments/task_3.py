def divide_numbers():
    try:
        first_number = float(input("Enter the first number: "))
        second_number = float(input("Enter the second number: "))
        result = first_number / second_number
    except ZeroDivisionError:
        print("Error: Cannot divide by zero.")
    except ValueError:
        print("Error: Invalid input. Please enter numeric values.")
    else:
        print(f"The result is {result}.")
    finally:
        print("This block always executes.")

divide_numbers()
