def factorial():
    number = int(input("Enter a number: "))
    result = 1
    for i in range(1, number + 1):
        result *= i
    print(f"The factorial of {number} is {result}.")

factorial()
