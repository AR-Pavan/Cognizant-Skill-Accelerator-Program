import turtle

# Recursive function to calculate the factorial of a number
def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)

# Recursive function to calculate the nth Fibonacci number
def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

# Recursive function to draw a fractal tree
def draw_fractal_tree(branch_length, t):
    if branch_length > 5:
        t.forward(branch_length)
        t.right(20)
        draw_fractal_tree(branch_length - 15, t)
        t.left(40)
        draw_fractal_tree(branch_length - 15, t)
        t.right(20)
        t.backward(branch_length)

# Function to display the menu and handle user input
def menu():
    while True:
        print("\nWelcome to the Recursive Artistry Program!")
        print("Choose an option:")
        print("1. Calculate Factorial")
        print("2. Find Fibonacci")
        print("3. Draw a Recursive Fractal")
        print("4. Exit")

        choice = input("> ")

        if choice == '1':
            try:
                num = int(input("Enter a number to find its factorial: "))
                if num < 0:
                    raise ValueError("Number must be non-negative.")
                print(f"The factorial of {num} is {factorial(num)}.")
            except ValueError as e:
                print(f"Invalid input: {e}")

        elif choice == '2':
            try:
                n = int(input("Enter the position of the Fibonacci number: "))
                if n < 0:
                    raise ValueError("Position must be non-negative.")
                print(f"The {n}th Fibonacci number is {fibonacci(n)}.")
            except ValueError as e:
                print(f"Invalid input: {e}")

        elif choice == '3':
            # Set up the turtle graphics window
            screen = turtle.Screen()
            screen.title("Recursive Fractal Tree")
            t = turtle.Turtle()
            t.left(90)
            t.speed(1)
            draw_fractal_tree(75, t)
            screen.mainloop()

        elif choice == '4':
            print("Exiting the program. Goodbye!")
            break

        else:
            print("Invalid choice. Please select a valid option.")

# Run the menu function to start the program
menu()
