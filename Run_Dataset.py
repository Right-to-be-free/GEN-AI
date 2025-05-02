import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

df = pd.read_csv(r"C:\Users\rishi\Desktop\Random forest\archive (4)\steps_tracker_dataset.csv")

# Set up a clean style for plots
sns.set(style="whitegrid")

# Create individual plots for trends and distributions
figures = {}

# Mood distribution
fig, ax = plt.subplots(figsize=(8, 5))
sns.countplot(data=df, x="mood", order=df["mood"].value_counts().index, ax=ax)
ax.set_title("Mood Distribution")
figures["Mood Distribution"] = fig

# Correlation heatmap
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.select_dtypes(include=["float64", "int64"]).corr(), annot=True, cmap="coolwarm", ax=ax)
ax.set_title("Correlation Heatmap")
figures["Correlation Heatmap"] = fig

# Time series trend: Steps over time
df['date'] = pd.to_datetime(df['date'], dayfirst=True)
df_sorted = df.sort_values("date")

fig, ax = plt.subplots(figsize=(12, 5))
sns.lineplot(data=df_sorted, x="date", y="steps", ax=ax)
ax.set_title("Steps Over Time")
figures["Steps Over Time"] = fig

# Water intake vs Sleep Hours
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(data=df, x="water_intake_liters", y="sleep_hours", hue="mood", ax=ax)
ax.set_title("Water Intake vs Sleep Hours")
figures["Water vs Sleep"] = fig

# Display all figures
figures
plt.show()
