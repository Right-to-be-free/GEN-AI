# coffee shop python program.
name = input("Enter customer name: ")
age = int(input("Enter customer age: "))
coffee_type = input("Enter coffee type (Espresso, Latte, Cappuccino): ")


# User enters: 4.99
price = eval(input("Enter coffee price: "))  
print(type(price))  # Output: <class 'float'>

# A list stores multiple coffee orders.
orders = ["Espresso", "Latte", "Cappuccino"]
print(orders[0])  # Output: Latte

 #String Manipulation - Formatting Customer Name
# ensure the name is capitalized properly.
formatted_name = name.title()  # Converts "john doe" → "John Doe"
print(f"Customer Name: {formatted_name}")

# Replace - Modifying Coffee Names
# If a customer accidentally orders "Expresso" instead of "Espresso," we auto-correct it.
coffee_type = coffee_type.replace("Expresso", "Espresso")
print(f"Updated Coffee Type: {coffee_type}")

# Even & Odd - Assigning Customer Loyalty Numbers
# even numbers get a special discount, odd numbers get a free cookie.

customer_id = eval(input("Enter customer ID number: "))

if customer_id % 2 == 0:
    print("You get a 10% discount! 🎉")
else:
    print("You get a free cookie! 🍪")

# Sets - Unique Customers of the Day
# We store unique customer names in a set (to avoid duplicates).
customers = {"Alice", "Bob", "Charlie"}  
customers.add(name)  # Adds a new customer if not already in the set
print(customers)  

# Dictionary - Storing Customer Order Details
# We store the order details in a dictionary.
customer_order = {
    "name": formatted_name,
    "age": age,
    "coffee": coffee_type,
    "price": price
}

print(customer_order)  




