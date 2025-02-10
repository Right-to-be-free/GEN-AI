import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from scipy.stats import norm

# ======================
# 1. Create Synthetic Dataset
# ======================
data = {
    'Rank': [1, 2, 3, 4, 5],
    'Video Name': ['See You Again', 'Shape of You', 'Despacito', 'Gangnam Style', 'Baby Shark'],
    'Views': [6547981039, 5470671677, 3471237000, 347123700, 1471237000],
    'Likes': [3497955, 19023293, 29356300, 7899000, 4567000],
    'Dislikes': [78799, 85900, 45678, 12345, 67890],
    'Category': ['Music', 'Music', 'Music', 'Entertainment', 'Education']
}

df = pd.DataFrame(data)

# Add viral flag using Bernoulli distribution threshold
df['Viral'] = np.where(df['Views'] > 1_000_000_000, 1, 0)

# ======================
# 2. Scalars and Vectors
# ======================
# Scalars
views_scalar = df.loc[df['Video Name'] == 'See You Again', 'Views'].values[0]
print(f"Scalar Example - Views for 'See You Again': {views_scalar:,}")

# Vectors
likes_vector = df['Likes'].values
dislikes_vector = df['Dislikes'].values
print(f"\nLikes Vector: {likes_vector}")
print(f"Dislikes Vector: {dislikes_vector}")

# ======================
# 3. Matrix Operations
# ======================
print("\nMatrix Representation:")
print(df[['Rank', 'Video Name', 'Views', 'Likes', 'Dislikes', 'Category']].head())

# ======================
# 4. Basis and PCA
# ======================
# Standardize numerical features
numeric_cols = ['Views', 'Likes', 'Dislikes']
X = df[numeric_cols].apply(lambda x: (x - x.mean()) / x.std())

# Perform PCA
pca = PCA(n_components=2)
principal_components = pca.fit_transform(X)

print("\nPrincipal Components (Eigenvectors):")
print(pca.components_)

# ======================
# 5. Central Tendency
# ======================
plt.figure(figsize=(15, 10))

# Mean, Median, Mode
plt.subplot(2, 2, 1)
sns.barplot(x=df['Category'].value_counts().index, 
            y=df['Category'].value_counts().values)
plt.title('Mode Analysis (Most Common Category)')

# ======================
# 6. Bernoulli Distribution
# ======================
plt.subplot(2, 2, 2)
sns.countplot(x='Viral', data=df)
plt.title('Bernoulli Distribution of Viral Videos')

# ======================
# 7. Normal Distribution
# ======================
plt.subplot(2, 2, 3)
views = df['Views'] / 1e9  # Convert to billions
mu, std = norm.fit(views)
sns.histplot(views, kde=True, stat='density')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k', linewidth=2)
plt.title(f'Normal Distribution Fit (μ={mu:.2f}B, σ={std:.2f})')
plt.xlabel('Views (Billions)')

# ======================
# 8. PCA Visualization
# ======================
plt.subplot(2, 2, 4)
plt.scatter(principal_components[:, 0], principal_components[:, 1], 
            c=df['Viral'], cmap='viridis')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Video Metrics')

plt.tight_layout()
plt.show()

# ======================
# Statistical Summary
# ======================
print("\nStatistical Summary:")
print(f"Mean Views: {df['Views'].mean():,.0f}")
print(f"Median Views: {df['Views'].median():,.0f}")
print(f"Mode Category: {df['Category'].mode()[0]}")
print(f"Viral Probability (Bernoulli p): {df['Viral'].mean():.2f}")