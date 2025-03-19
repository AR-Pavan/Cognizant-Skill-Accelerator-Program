# Ask the user to input a password
password = input("Enter a password: ")

# Initialize flags for each check
has_upper = False
has_lower = False
has_digit = False
has_special = False

# Check the length of the password
if len(password) >= 8:
    length_check = True
else:
    length_check = False

# Check for uppercase letter
for char in password:
    if char.isupper():
        has_upper = True
        break

# Check for lowercase letter
for char in password:
    if char.islower():
        has_lower = True
        break

# Check for digit
for char in password:
    if char.isdigit():
        has_digit = True
        break

# Check for special character
special_characters = "!@#$%^&*()-_+=<>?{}[]|:;',./`~"
for char in password:
    if char in special_characters:
        has_special = True
        break

# Provide feedback based on the checks
if length_check and has_upper and has_lower and has_digit and has_special:
    print("Your password is strong! 💪")
else:
    if not length_check:
        print("Your password needs to be at least 8 characters long.😔")
    if not has_upper:
        print("Your password needs at least one uppercase letter.😔")
    if not has_lower:
        print("Your password needs at least one lowercase letter.😔")
    if not has_digit:
        print("Your password needs at least one digit.😔")
    if not has_special:
        print("Your password needs at least one special character.😔")
# Calculate the strength score
strength_score = 0
if length_check:
    strength_score += 2
if has_upper:
    strength_score += 2
if has_lower:
    strength_score += 2
if has_digit:
    strength_score += 2
if has_special:
    strength_score += 2

# Print the strength score
print(f"Password Strength Score: {strength_score}/10")
