import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import bernoulli, binom, norm

# 1. Bernoulli Distribution: Will a Customer Buy?
# Probability of buying organic apples (p = 0.4)
p_buy = 0.4
bernoulli_trials = bernoulli.rvs(p_buy, size=1000)  # Simulating 1000 customers

# Visualizing Bernoulli Distribution
plt.figure(figsize=(6, 4))
plt.hist(bernoulli_trials, bins=2, color='skyblue', rwidth=0.8)
plt.title('Bernoulli Distribution: Will a Customer Buy Organic Apples?')
plt.xlabel('Outcome (0 = No Buy, 1 = Buy)')
plt.ylabel('Frequency')
plt.xticks([0, 1])
plt.show()

# 2. Binomial Distribution: Predicting Sales
n_customers = 100  # Number of customers visiting the store daily
p_buy_apples = 0.4  # Probability of buying apples
binomial_sales = binom.rvs(n=n_customers, p=p_buy_apples, size=1000)

# Visualizing Binomial Distribution
plt.figure(figsize=(8, 5))
plt.hist(binomial_sales, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
plt.title('Binomial Distribution: Daily Sales of Organic Apples')
plt.xlabel('Number of Sales')
plt.ylabel('Frequency')
plt.show()

# 3. Normal Distribution: Daily Milk Sales
mu_milk = 200  # Mean daily milk sales (gallons)
sigma_milk = 20  # Standard deviation
milk_sales = norm.rvs(loc=mu_milk, scale=sigma_milk, size=1000)

# Visualizing Normal Distribution
plt.figure(figsize=(8, 5))
plt.hist(milk_sales, bins=30, color='orange', edgecolor='black', alpha=0.7)
plt.title('Normal Distribution: Daily Milk Sales')
plt.xlabel('Milk Sales (Gallons)')
plt.ylabel('Frequency')
plt.axvline(mu_milk, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mu_milk}')
plt.legend()
plt.show()

# 4. Mean, Median, and Mode: Summarizing Data
revenue_data = [5000, 4800, 4500, 5200, 4500, 4700] * 20
mean_revenue = np.mean(revenue_data)
median_revenue = np.median(revenue_data)
mode_revenue = pd.Series(revenue_data).mode()[0]

print(f"Mean Revenue: ${mean_revenue}")
print(f"Median Revenue: ${median_revenue}")
print(f"Mode Revenue: ${mode_revenue}")

# Visualizing Revenue Summary
plt.figure(figsize=(6, 4))
plt.boxplot(revenue_data)
plt.title('Revenue Summary (Mean, Median, Mode)')
plt.ylabel('Revenue ($)')
plt.show()

# 5. Vector Spaces: Customer Preferences
# Representing customer preferences as vectors
customers_preferences = np.array([
    [3, 5, 2, 7],   # [Fruits, Vegetables, Snacks, Dairy]
    [4, 6, 3, 8],
    [2, 4, 1, 5],
])
scaled_preferences = customers_preferences * np.array([1.2, 1.1, 1.5, 1])  # Scaling snacks for promotions

print("Original Preferences:\n", customers_preferences)
print("Scaled Preferences:\n", scaled_preferences)

# Visualizing Customer Preferences Before and After Scaling
categories = ['Fruits', 'Vegetables', 'Snacks', 'Dairy']
original_avg = customers_preferences.mean(axis=0)
scaled_avg = scaled_preferences.mean(axis=0)

x = np.arange(len(categories))
width = 0.35

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - width/2, original_avg, width, label='Original', color='blue')
ax.bar(x + width/2, scaled_avg, width, label='Scaled', color='green')

ax.set_xlabel('Categories')
ax.set_ylabel('Average Preference Score')
ax.set_title('Customer Preferences Before and After Scaling')
ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.legend()
plt.show()

# Python File Read/Write Example
# Writing sales data to a CSV file
sales_data = pd.DataFrame({
    'Product': ['Milk', 'Bread', 'Eggs'],
    'Sales': [2000, 1500, 1200]
})
sales_data.to_csv('sales_data.csv', index=False)

# Reading the file back
read_sales_data = pd.read_csv('sales_data.csv')
print("Sales Data from CSV:\n", read_sales_data)

# Sets for Unique Customers
customers_today = {"Alice", "Bob", "Charlie"}
customers_yesterday = {"Alice", "David"}
unique_customers = customers_today.union(customers_yesterday)

print("Unique Customers:", unique_customers)
