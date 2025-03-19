# Task 1

name = "Thor Odinson"
age = "1500"
height = "6.6"

print("Hey There! My name is "+name+", Asgardian God of Thunder. I am "+age+" and my height is "+height+" feet.")


# task 2

num1 = 10
num2 = 5
# for ADD the numbers
add = num1 + num2 
# for Subtracting the numbers
sub = num1 - num2
# for multiplying the numbers
mul = num1*num2
# for dividing the numbers
div = num1/num2
# printing the results
print("addition of numbers",num1,"and",num2,"is",add)
print("Substracting",num2,"from",num1,"gives us",sub)
print("multiplying",num1,"with",num2,"gives us",mul)
print("dividing",num1,"with",num2,"gives us",int(div))

# Task 3

print("Enter the number:")
num = int(input())
if num>0:
    print("This number is positive. Awesome!")
elif num<0:
    print("This number is negative. Better luck next time!")
else:
    print("Zero it is. A perfect balance!")
