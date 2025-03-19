# Create a tuple
favorites = ('Inception', 'Bohemian Rhapsody', '1984')

# Print the tuple
print("Favorite things:", favorites)

# Attempt to change one of the elements
try:
    favorites[0] = 'Interstellar'
except TypeError:
    print("Oops! Tuples cannot be changed.")

# Print the length of the tuple
print("Length of tuple:", len(favorites))
