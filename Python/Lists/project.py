# Initialize an empty inventory dictionary
inventory = {
    "apple": (10, 2.5),
    "banana": (20, 1.2)
}

# Function to add a new item to the inventory
def add_item(name, quantity, price):
    inventory[name] = (quantity, price)

# Function to remove an item from the inventory
def remove_item(name):
    if name in inventory:
        del inventory[name]
    else:
        print(f"Item '{name}' not found in inventory.")

# Function to update the quantity or price of an existing item
def update_item(name, quantity=None, price=None):
    if name in inventory:
        current_quantity, current_price = inventory[name]
        if quantity is not None:
            current_quantity = quantity
        if price is not None:
            current_price = price
        inventory[name] = (current_quantity, current_price)
    else:
        print(f"Item '{name}' not found in inventory.")

# Function to display the inventory
def display_inventory():
    print("Current inventory:")
    for item, (quantity, price) in inventory.items():
        print(f"Item: {item}, Quantity: {quantity}, Price: ${price:.2f}")

# Function to calculate the total value of the inventory
def calculate_total_value():
    total_value = sum(quantity * price for quantity, price in inventory.values())
    return total_value

# Example run
print("Welcome to the Inventory Manager!")

# Display initial inventory
display_inventory()

# Ask the user to add items
num_items = int(input("How many items would you like to add? "))
for _ in range(num_items):
    name = input("Enter the item name: ")
    quantity = int(input("Enter the quantity: "))
    price = float(input("Enter the price: "))
    add_item(name, quantity, price)

# Display updated inventory
print("\nUpdated inventory:")
display_inventory()

# Calculate and display the total value of the inventory
total_value = calculate_total_value()
print(f"\nTotal value of inventory: ${total_value:.2f}")
