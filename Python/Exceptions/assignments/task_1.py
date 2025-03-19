def divide_hundred():
    try:
        number = input("Enter a number: ")
        number = float(number)
        result = 100 / number
        print(f"100 divided by {number} is {result}.")
    except ZeroDivisionError:
        print("Oops! You cannot divide by zero.")
    except ValueError:
        print("Invalid input! Please enter a valid number.")

divide_hundred()
