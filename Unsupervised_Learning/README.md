Unsupervised Learning Guide
What is Unsupervised Learning?
**Definition**: Unsupervised Learning is a type of machine learning where the model identifies patterns in data without labeled outcomes.
**Analogy**: Like a person organizing files without knowing their labels—grouping based on similarities.
```python
# Example: clustering with KMeans
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(X)
```
Why Unsupervised Learning?
Used when labeled data is unavailable or expensive to obtain.
Ideal for pattern discovery, customer segmentation, and anomaly detection.
Key Concepts
- **Clustering**: Grouping similar data points together (e.g., K-Means, DBSCAN).
- **Dimensionality Reduction**: Reducing features while preserving variance (e.g., PCA, t-SNE).
- **Association Rule Mining**: Discovering interesting relations (e.g., Apriori).
Clustering Techniques
- **K-Means**: Partitions data into k clusters based on proximity.
- **DBSCAN**: Groups based on density, useful for noisy data.
- **Hierarchical Clustering**: Builds a tree of clusters (dendrogram).
```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
```
Dimensionality Reduction
- **PCA**: Projects data into fewer dimensions by preserving variance.
- **t-SNE**: For visualizing high-dimensional data in 2D/3D.
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
```
Association Rule Mining
- Finds frequent itemsets and strong rules (used in market basket analysis).
```python
from mlxtend.frequent_patterns import apriori
frequent = apriori(df, min_support=0.1, use_colnames=True)
```
Applications of Unsupervised Learning
- Customer segmentation in marketing.
- Anomaly detection in fraud or network security.
- Recommendation systems.
- Document/topic clustering.
- Image compression and recognition.
Model Evaluation Techniques
- **Silhouette Score**: Measures cluster quality.
- **Elbow Method**: Helps choose number of clusters.
```python
from sklearn.metrics import silhouette_score
score = silhouette_score(X, kmeans.labels_)
```
Challenges
- No ground truth for validation.
- Interpreting clusters may be subjective.
- Scaling and preprocessing often critical.
Best Practices
- Visualize clusters for insight.
- Try multiple clustering algorithms.
- Normalize and scale data beforehand.
- Use dimensionality reduction for large feature sets.
Common Interview Questions
- How does K-Means clustering work?
- What is the difference between PCA and t-SNE?
- How do you evaluate clustering performance?
- Give real-world use cases of unsupervised learning.

- ---

Unsupervised Learning - Real Life Examples
1. Customer Segmentation for Marketing
Problem
A retail company wants to segment its customers to run personalized marketing campaigns.
Solution
Use K-Means clustering on features like purchase frequency, average order value, and recency.
Implementation
```python
from sklearn.cluster import KMeans
model = KMeans(n_clusters=4)
model.fit(customer_data)
```
Impact
The company tailored campaigns to each segment, improving conversion rates by 25%.
Conclusion
Clustering allowed the business to uncover natural groupings without needing labeled data.
2. Anomaly Detection in Credit Card Transactions
Problem
A bank wants to detect suspicious (fraudulent) transactions without prior examples of fraud.
Solution
Use DBSCAN to detect outliers in transaction patterns.
Implementation
```python
from sklearn.cluster import DBSCAN
db = DBSCAN(eps=0.5, min_samples=5).fit(transactions)
```
Impact
Enabled real-time flagging of anomalous behavior with a 15% reduction in false positives.
Conclusion
Unsupervised learning identified hidden fraud patterns with minimal labeled examples.
3. Document Clustering for News Categorization
Problem
A news aggregator wants to group articles by topic without predefined categories.
Solution
Use TF-IDF vectorization followed by KMeans clustering.
Implementation
```python
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(news_texts)
```
Impact
Improved topic-based navigation and user personalization.
Conclusion
Text clustering enhanced content discoverability and reading experience.
4. Product Recommendations via Association Rules
Problem
An e-commerce site wants to suggest items frequently bought together.
Solution
Use Apriori algorithm to find frequent itemsets.
Implementation
```python
from mlxtend.frequent_patterns import apriori
frequent_itemsets = apriori(basket_data, min_support=0.1, use_colnames=True)
```
Impact
Recommended combos increased cart value by 20%.
Conclusion
Unsupervised rules mining revealed relationships without explicit labeling.
5. Image Compression using PCA
Problem
Reduce storage cost by compressing high-resolution image datasets.
Solution
Apply PCA to reduce the number of dimensions in pixel data.
Implementation
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=50)
compressed = pca.fit_transform(images)
```
Impact
Achieved ~80% reduction in storage with minimal quality loss.
Conclusion
Dimensionality reduction helped manage large visual datasets efficiently.

