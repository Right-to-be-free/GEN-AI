import matplotlib.pyplot as plt

# Data
measures = ['Mean', 'Median', 'Mode']
values = [5000, 4800, 4500]
colors = ['#4C72B0', '#55A868', '#C44E52']

# Plot
plt.figure(figsize=(8, 5))
bars = plt.bar(measures, values, color=colors, edgecolor='black')

# Customize
plt.title("Daily Revenue Central Tendency", pad=20, fontsize=14, fontweight='bold')
plt.ylabel("USD ($)", fontsize=12)
plt.ylim(4000, 5100)

# Add value labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, height + 20,
             f'${height:,}', ha='center', va='bottom', fontsize=11)

# Remove unnecessary chart junk
plt.gca().spines[['top', 'right']].set_visible(False)
plt.grid(axis='y', linestyle='--', alpha=0.4)

plt.tight_layout()
plt.show()