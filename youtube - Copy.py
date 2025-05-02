import os
import pandas as pd

# Define the path to the directory containing the CSV file
extract_path = "C:/Users/rishi/Desktop/Gen Ai"

# Load the CSV file into a DataFrame
csv_file_path = os.path.join(extract_path, "Most popular 1000 Youtube videos.csv")
df = pd.read_csv(csv_file_path)

# Display basic information about the dataset
df.info()
df.head()
