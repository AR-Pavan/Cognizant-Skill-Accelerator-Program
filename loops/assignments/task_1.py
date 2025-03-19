
def countdown():
    start = int(input("Enter the starting number: "))
    while start > 0:
        print(start, end=' ')
        start -= 1
    print("Blast off! 🚀")

countdown()
