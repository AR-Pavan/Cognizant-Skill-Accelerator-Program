# Create a dictionary with personal information
info = {"name": "Alice", "age": 25, "city": "New York"}

# Add a new key-value pair for favorite color
info["favorite color"] = "Blue"

# Update the city key with a new value
info["city"] = "Los Angeles"

# Print all keys and values
print("Keys:", ", ".join(info.keys()))
print("Values:", ", ".join(str(value) for value in info.values()))
