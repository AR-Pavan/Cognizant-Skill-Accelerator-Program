print("Please enter your age:")
age = int(input())
# changed the type of age from string to intiger
if age<0:
    print("Please enter the valid age")
else:
    if age>18:
        print(" Congratulations! You are eligible to vote. Go make a difference!")
    else:
        print("Oops! You’re not eligible yet. But hey, only",18-age, "more years to go!")
