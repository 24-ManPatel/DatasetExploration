import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import silhouette_score
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

# Load the data
# Note: In a real scenario, you'd replace this with your actual file path
df = pd.read_csv('ChicagoCrimeData.csv')
# For demonstration, I'll create a sample dataframe with the structure shown

# In a real scenario, you'd use your actual CSV file
# This is just to show the structure of the code that would work on your full dataset

# Basic data preprocessing
print("Data Overview:")
print(f"Shape: {df.shape}")
print("\nData Types:")
print(df.dtypes)
print("\nMissing Values:")
print(df.isnull().sum())

# Convert DATE to datetime
df['DATE'] = pd.to_datetime(df['DATE'])
df['HOUR'] = df['DATE'].dt.hour
df['DAY'] = df['DATE'].dt.day
df['MONTH'] = df['DATE'].dt.month
df['DAY_OF_WEEK'] = df['DATE'].dt.dayofweek

# For demonstration, let's add more synthetic data to show the models working
# In real application, you would skip this step and use your full dataset
np.random.seed(42)
additional_rows = 500
synthetic_data = {
    'ID': np.random.randint(1000000, 5000000, additional_rows),
    'PRIMARY_TYPE': np.random.choice(['THEFT', 'ASSAULT', 'BURGLARY', 'ROBBERY'], additional_rows),
    'HOUR': np.random.randint(0, 24, additional_rows),
    'DAY': np.random.randint(1, 31, additional_rows),
    'MONTH': np.random.randint(1, 13, additional_rows),
    'DAY_OF_WEEK': np.random.randint(0, 7, additional_rows),
    'ARREST': np.random.choice([True, False], additional_rows),
    'DISTRICT': np.random.randint(1, 20, additional_rows),
    'COMMUNITY_AREA_NUMBER': np.random.randint(1, 80, additional_rows),
    'LATITUDE': np.random.uniform(41.7, 42.0, additional_rows),
    'LONGITUDE': np.random.uniform(-87.9, -87.5, additional_rows)
}
synthetic_df = pd.DataFrame(synthetic_data)
df_extended = pd.concat([df, synthetic_df], ignore_index=True)

print("\nExtended dataset shape:", df_extended.shape)

# EDA: Plot crime distribution by type
plt.figure(figsize=(10, 6))
crime_counts = df_extended['PRIMARY_TYPE'].value_counts().head(10)
sns.barplot(x=crime_counts.index, y=crime_counts.values)
plt.title('Top 10 Crime Types')
plt.xlabel('Crime Type')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('crime_distribution.png')

# EDA: Plot crimes by hour of day
plt.figure(figsize=(10, 6))
sns.countplot(data=df_extended, x='HOUR')
plt.title('Crimes by Hour of Day')
plt.xlabel('Hour')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('crimes_by_hour.png')

# EDA: Plot crime locations on a map
plt.figure(figsize=(10, 8))
plt.scatter(df_extended['LONGITUDE'], df_extended['LATITUDE'], alpha=0.5, s=10)
plt.title('Crime Locations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.tight_layout()
plt.savefig('crime_locations.png')

# Model 1: Random Forest for Arrest Prediction
print("\n--- Model 1: Random Forest for Arrest Prediction ---")

# Select features and target
X = df_extended[['PRIMARY_TYPE', 'HOUR', 'DAY', 'MONTH', 'DAY_OF_WEEK', 'DISTRICT', 'COMMUNITY_AREA_NUMBER']]
y = df_extended['ARREST']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Preprocessing pipeline
categorical_features = ['PRIMARY_TYPE']
numerical_features = ['HOUR', 'DAY', 'MONTH', 'DAY_OF_WEEK', 'DISTRICT', 'COMMUNITY_AREA_NUMBER']

# Create transformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numerical_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    ])

# Create and train the model
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

rf_pipeline.fit(X_train, y_train)

# Make predictions
y_pred = rf_pipeline.predict(X_test)

# Evaluate the model
print("\nRandom Forest Model Evaluation:")
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Plot confusion matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig('confusion_matrix.png')

# Feature importance
if hasattr(rf_pipeline[-1], 'feature_importances_'):
    # Get feature names
    cat_features = rf_pipeline[0].transformers_[1][1][-1].get_feature_names_out(categorical_features)
    all_features = list(numerical_features) + list(cat_features)
    
    # Get feature importances
    importances = rf_pipeline[-1].feature_importances_
    
    # Plot feature importances
    plt.figure(figsize=(10, 6))
    indices = np.argsort(importances)[-10:]  # Get top 10 features
    plt.barh(range(len(indices)), importances[indices])
    plt.yticks(range(len(indices)), [all_features[i] for i in indices])
    plt.title('Top 10 Feature Importances - Random Forest')
    plt.tight_layout()
    plt.savefig('feature_importance.png')

# Model 2: K-Means Clustering for Crime Hotspot Identification
print("\n--- Model 2: K-Means Clustering for Crime Hotspot Identification ---")

# Select features for clustering
X_cluster = df_extended[['LATITUDE', 'LONGITUDE']]
X_cluster = X_cluster.dropna()

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_cluster)

# Find optimal number of clusters using silhouette score
silhouette_scores = []
k_range = range(2, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(score)
    print(f"K = {k}, Silhouette Score = {score:.4f}")

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(k_range, silhouette_scores, 'bo-')
plt.title('Silhouette Score by Number of Clusters')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.grid(True)
plt.tight_layout()
plt.savefig('silhouette_scores.png')

# Choose the best k (for this example, we'll use the one with highest silhouette score)
best_k = k_range[np.argmax(silhouette_scores)]
print(f"\nBest number of clusters based on silhouette score: {best_k}")

# Apply KMeans with the best k
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_scaled)

# Add cluster labels to the original data
df_cluster = X_cluster.copy()
df_cluster['Cluster'] = cluster_labels

# Visualize the clusters
plt.figure(figsize=(12, 8))
for cluster in range(best_k):
    cluster_data = df_cluster[df_cluster['Cluster'] == cluster]
    plt.scatter(cluster_data['LONGITUDE'], cluster_data['LATITUDE'], 
                label=f'Cluster {cluster}', alpha=0.7)

# Plot cluster centers
centers = scaler.inverse_transform(kmeans.cluster_centers_)
plt.scatter(centers[:, 1], centers[:, 0], c='black', s=200, alpha=0.7, marker='X', label='Cluster Centers')
plt.title(f'Crime Hotspots - KMeans Clustering (k={best_k})')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('crime_clusters.png')

# Analyze clusters
print("\nCluster Analysis:")
cluster_stats = df_cluster.groupby('Cluster').size().reset_index(name='Count')
cluster_stats['Percentage'] = cluster_stats['Count'] / cluster_stats['Count'].sum() * 100
print(cluster_stats)

print("\nAnalysis completed!")