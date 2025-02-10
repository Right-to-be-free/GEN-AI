# Taking customer details
name = input("Enter customer name: ").title()
age = int(input("Enter customer age: "))
coffee_type = input("Enter coffee type (Espresso, Latte, Cappuccino): ")
price = eval(input("Enter coffee price: "))

# Correcting misspelled coffee type
coffee_type = coffee_type.replace("Expresso", "Espresso")

# Assigning loyalty discount
customer_id = eval(input("Enter customer ID number: "))
if customer_id % 2 == 0:
    print("You get a 10% discount! 🎉")
else:
    print("You get a free cookie! 🍪")

# Storing unique customers
customers = {"Alice", "Bob", "Charlie"}
customers.add(name)

# Creating a dictionary for the order
customer_order = {
    "name": name,
    "age": age,
    "coffee": coffee_type,
    "price": price
}

# Printing customer order
print("\nOrder Summary:")
print(customer_order)