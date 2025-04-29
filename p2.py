import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import silhouette_score, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.impute import SimpleImputer
import folium
from folium.plugins import HeatMap, MarkerCluster
import calendar
from scipy import stats
import matplotlib.dates as mdates
import warnings
warnings.filterwarnings('ignore')

# Load the data
print("Loading the crime dataset...")
# Load your CSV file
df = pd.read_csv('ChicagoCrimeData.csv')  # Replace with your actual file path

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
df['WEEKDAY'] = df['DATE'].dt.day_name()
df['YEAR'] = df['DATE'].dt.year

if df['ARREST'].dtype != bool:
    df['ARREST'] = df['ARREST'].map({'true': True, 'false': False, 'TRUE': True, 'FALSE': False})


# 1. Crime Types Distribution
plt.figure(figsize=(12, 8))
crime_counts = df['PRIMARY_TYPE'].value_counts().head(10)
total_crimes = crime_counts.sum()
ax = sns.barplot(x=crime_counts.index, y=crime_counts.values, palette='viridis')
plt.title('Top 10 Crime Types', fontsize=16)
plt.xlabel('Crime Type', fontsize=14)
plt.ylabel('Count', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)

# Add percentage labels on bars
for i, count in enumerate(crime_counts.values):
    percentage = count / total_crimes * 100
    ax.text(i, count + 5, f'{percentage:.1f}%', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('crime_distribution.png')
print("Saved crime distribution plot.")

# 2. Crimes by Hour
plt.figure(figsize=(14, 8))
hourly_crimes = df.groupby('HOUR').size().reset_index(name='Count')
sns.lineplot(data=hourly_crimes, x='HOUR', y='Count', marker='o', linewidth=2, color='steelblue')

# Add a moving average trendline if there's enough data
if len(hourly_crimes) > 3:
    plt.plot(hourly_crimes['HOUR'], hourly_crimes['Count'].rolling(window=3, center=True).mean(), 
             'r--', linewidth=2, label='3-hour Moving Average')
    plt.legend()

plt.title('Crimes by Hour of Day', fontsize=16)
plt.xlabel('Hour of Day (24-hour format)', fontsize=14)
plt.ylabel('Number of Crimes', fontsize=14)
plt.xticks(range(0, 24))
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('crimes_by_hour.png')
print("Saved crimes by hour plot.")

# 3. Crimes by Day of Week
plt.figure(figsize=(14, 8))
day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_crimes = df['WEEKDAY'].value_counts().reindex(day_order)
sns.barplot(x=day_crimes.index, y=day_crimes.values, palette='muted')
plt.title('Crimes by Day of Week', fontsize=16)
plt.xlabel('Day of Week', fontsize=14)
plt.ylabel('Number of Crimes', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('crimes_by_day.png')
print("Saved crimes by day plot.")

# 4. Monthly Crime Trends
plt.figure(figsize=(14, 8))
# Group by year and month to see trends over time
df['YearMonth'] = df['DATE'].dt.to_period('M')
monthly_crimes = df.groupby('YearMonth').size()
monthly_crimes.index = monthly_crimes.index.to_timestamp()

plt.plot(monthly_crimes.index, monthly_crimes.values, marker='o', linestyle='-', linewidth=2)
plt.title('Monthly Crime Trends', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('Number of Crimes', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gca().xaxis.set_major_locator(mdates.YearLocator())
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('monthly_crime_trends.png')
print("Saved monthly crime trends plot.")

# 5. Arrest Rate by Crime Type
plt.figure(figsize=(14, 8))
arrest_by_type = df.groupby('PRIMARY_TYPE')['ARREST'].mean().sort_values(ascending=False).head(10)
sns.barplot(x=arrest_by_type.index, y=arrest_by_type.values, palette='coolwarm')
plt.title('Arrest Rate by Crime Type (Top 10)', fontsize=16)
plt.xlabel('Crime Type', fontsize=14)
plt.ylabel('Arrest Rate', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('arrest_rate_by_crime.png')
print("Saved arrest rate by crime type plot.")

# 6. Crime Locations Map (if coordinates are available and not null)
if 'LATITUDE' in df.columns and 'LONGITUDE' in df.columns and not df[['LATITUDE', 'LONGITUDE']].isna().all().any():
    # Filter out rows with missing coordinates
    map_df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
    
    # Create a sample for the map if dataset is large
    if len(map_df) > 5000:
        map_sample = map_df.sample(5000, random_state=42)
    else:
        map_sample = map_df
        
    # Create a base map centered around the mean coordinates
    base_map = folium.Map(
        location=[map_sample['LATITUDE'].mean(), map_sample['LONGITUDE'].mean()],
        zoom_start=11
    )

    # Add a heatmap layer
    heat_data = [[row['LATITUDE'], row['LONGITUDE']] for _, row in map_sample.iterrows()]
    HeatMap(heat_data, radius=15).add_to(base_map)
    
    # Save the map
    base_map.save('crime_heatmap.html')
    print("Saved crime heatmap.")
    
    # Create a clustered marker map
    marker_map = folium.Map(
        location=[map_sample['LATITUDE'].mean(), map_sample['LONGITUDE'].mean()],
        zoom_start=11
    )
    
    # Create a marker cluster
    marker_cluster = MarkerCluster().add_to(marker_map)
    
    # Add markers for a subset of the data
    for idx, row in map_sample.sample(min(1000, len(map_sample))).iterrows():
        popup_text = f"Type: {row['PRIMARY_TYPE']}<br>Description: {row['DESCRIPTION']}<br>Arrest: {row['ARREST']}"
        folium.Marker(
            location=[row['LATITUDE'], row['LONGITUDE']],
            popup=popup_text
        ).add_to(marker_cluster)
    
    # Save the marker map
    marker_map.save('crime_markers.html')
    print("Saved crime marker map.")
else:
    print("Skipping map creation - latitude or longitude data is missing.")


# MODEL 1: RANDOM FOREST FOR ARREST PREDICTION

print("\n--- MODEL 1: RANDOM FOREST FOR ARREST PREDICTION ---")

# Choose relevant features for prediction
features = ['PRIMARY_TYPE', 'LOCATION_DESCRIPTION', 'HOUR', 'DAY_OF_WEEK', 'MONTH', 
            'DISTRICT', 'WARD', 'COMMUNITY_AREA_NUMBER']

# Filter to only include rows that have these features and the target variable
model_df = df.copy()
for feature in features + ['ARREST']:
    if feature in model_df.columns:
        model_df = model_df[model_df[feature].notna()]
    else:
        print(f"Warning: Feature '{feature}' not found in dataset. Removing from model.")
        features.remove(feature)

# Check if we have enough data for modeling
if len(model_df) < 100:
    print("Not enough data for modeling after filtering. Please check your dataset.")
else:
    # Select features and target
    X = model_df[features]
    y = model_df['ARREST']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Identify categorical and numerical features
    categorical_features = [feat for feat in features if X[feat].dtype == 'object' or X[feat].dtype.name == 'category']
    numerical_features = [feat for feat in features if feat not in categorical_features]
    
    print(f"Categorical features: {categorical_features}")
    print(f"Numerical features: {numerical_features}")
    
    # Create preprocessing pipeline
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
    
    # Train the model
    print("Training Random Forest model...")
    rf_pipeline.fit(X_train, y_train)
    
    # Make predictions
    y_pred = rf_pipeline.predict(X_test)
    y_pred_proba = rf_pipeline.predict_proba(X_test)[:, 1]
    
    # Evaluate 
    print("\nRandom Forest Model Evaluation:")
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix', fontsize=16)
    plt.ylabel('Actual', fontsize=14)
    plt.xlabel('Predicted', fontsize=14)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    print("Saved confusion matrix plot.")
    
    # ROC Curve
    plt.figure(figsize=(10, 8))
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('roc_curve.png')
    print("Saved ROC curve plot.")
    
    # Precision-Recall Curve
    plt.figure(figsize=(10, 8))
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.plot(recall_curve, precision_curve, color='darkgreen', lw=2)
    plt.xlabel('Recall', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.title('Precision-Recall Curve', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('precision_recall_curve.png')
    print("Saved precision-recall curve plot.")
    
    # Hyperparameter Tuning for Random Forest (if dataset is not too large)
    if len(X_train) < 10000:  # Only perform if dataset is manageable
        print("\nPerforming hyperparameter tuning for Random Forest...")
        param_grid = {
            'classifier__n_estimators': [50, 100],
            'classifier__max_depth': [None, 10, 20],
            'classifier__min_samples_split': [2, 5]
        }
        
        grid_search = GridSearchCV(
            rf_pipeline,
            param_grid,
            cv=3,
            scoring='f1',
            n_jobs=2  # Adjust based on your machine's capabilities
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best F1 score: {grid_search.best_score_:.4f}")
        
        # Use the best model
        best_rf_model = grid_search.best_estimator_
    else:
        best_rf_model = rf_pipeline
    
    # Function to predict arrest for new crime data
    def predict_arrest(crime_data):
        """
        Predict whether an arrest will be made for a given crime incident
        
        Parameters:
        crime_data (dict): Dictionary containing crime details matching the feature names
        
        Returns:
        tuple: (prediction, probability)
        """
        # Create a DataFrame with the crime data
        crime_df = pd.DataFrame([crime_data])
        
        # Ensure all required features are present
        for feature in features:
            if feature not in crime_data:
                crime_df[feature] = None  # will be imputed in the pipeline
        
        # Make prediction
        prediction = best_rf_model.predict(crime_df[features])[0]
        probability = best_rf_model.predict_proba(crime_df[features])[0, 1]
        
        return prediction, probability
    
    # Feature importance visualization (if Random Forest is used)
    if hasattr(best_rf_model[-1], 'feature_importances_'):
        print("\nAnalyzing feature importance...")
        
        # Get the preprocessor and model
        preprocessor = best_rf_model.named_steps['preprocessor']
        rf_classifier = best_rf_model.named_steps['classifier']
        
        # Get feature names after one-hot encoding
        feature_names = []
        
        # Add numerical feature names
        if numerical_features:
            feature_names.extend(numerical_features)
        
        # Add categorical feature names (after one-hot encoding)
        if categorical_features:
            cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
            cat_feature_names = []
            # Check if the encoder has been fit
            if hasattr(cat_encoder, 'get_feature_names_out'):
                cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
                feature_names.extend(cat_feature_names)
        
        # Get feature importances
        importances = rf_classifier.feature_importances_
        
        # Map importances to feature names
        if len(feature_names) == len(importances):
            # Create a dataframe of feature importances
            feature_imp = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values(by='Importance', ascending=False)
            
            # Plot top 20 features (or all if less than 20)
            top_n = min(20, len(feature_imp))
            plt.figure(figsize=(12, 10))
            sns.barplot(x='Importance', y='Feature', data=feature_imp.head(top_n))
            plt.title(f'Top {top_n} Feature Importances for Arrest Prediction', fontsize=16)
            plt.xlabel('Relative Importance', fontsize=14)
            plt.tight_layout()
            plt.savefig('feature_importance.png')
            print(f"Saved top {top_n} feature importance plot.")
            
            # Print top 10 features
            print("\nTop 10 most important features:")
            for i, (feature, importance) in enumerate(zip(feature_imp['Feature'].head(10), feature_imp['Importance'].head(10))):
                print(f"{i+1}. {feature}: {importance:.4f}")
        else:
            print("Warning: Number of feature names doesn't match importances. Skipping feature importance visualization.")
    
    # Example of using the prediction function
    print("\nExample: Predicting arrest for a sample crime")
    
    # Get a sample from the test set
    sample = X_test.iloc[0].to_dict()
    
    print("Sample crime details:")
    for key, value in sample.items():
        print(f"  {key}: {value}")
    
    arrest_pred, arrest_prob = predict_arrest(sample)
    print(f"\nPrediction: {'Arrest' if arrest_pred else 'No Arrest'}")
    print(f"Probability of arrest: {arrest_prob:.2f}")
    
    # Calculate arrest rate by location
    if 'LOCATION_DESCRIPTION' in df.columns:
        print("\nAnalyzing arrest rates by location...")
        location_arrest_rates = df.groupby('LOCATION_DESCRIPTION')['ARREST'].agg(['mean', 'count'])
        location_arrest_rates.columns = ['Arrest Rate', 'Count']
        
        # Filter to locations with at least 10 incidents
        location_arrest_rates = location_arrest_rates[location_arrest_rates['Count'] >= 10]
        location_arrest_rates = location_arrest_rates.sort_values('Arrest Rate', ascending=False)
        
        # Plot arrest rates by top locations
        plt.figure(figsize=(14, 10))
        top_locations = location_arrest_rates.head(15)
        sns.barplot(x='Arrest Rate', y=top_locations.index, data=top_locations)
        plt.title('Arrest Rates by Location (Top 15 locations)', fontsize=16)
        plt.xlabel('Arrest Rate', fontsize=14)
        plt.ylabel('Location', fontsize=14)
        plt.xlim(0, 1)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('arrest_rate_by_location.png')
        print("Saved arrest rate by location plot.")


# MODEL 2: K-MEANS CLUSTERING FOR CRIME HOTSPOT IDENTIFICATION

print("\n--- MODEL 2: K-MEANS CLUSTERING FOR CRIME HOTSPOT IDENTIFICATION ---")

# Check if we have geographic coordinates
if 'LATITUDE' in df.columns and 'LONGITUDE' in df.columns and not df[['LATITUDE', 'LONGITUDE']].isna().all().any():
    # Filter out rows with missing coordinates
    geo_df = df.dropna(subset=['LATITUDE', 'LONGITUDE'])
    
    # Select features for clustering
    X_cluster = geo_df[['LATITUDE', 'LONGITUDE']]
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    # Find optimal number of clusters using silhouette score
    silhouette_scores = []
    k_range = range(2, 11)
    
    print("Evaluating optimal number of clusters...")
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_scaled)
        score = silhouette_score(X_scaled, kmeans.labels_)
        silhouette_scores.append(score)
        print(f"K = {k}, Silhouette Score = {score:.4f}")
    
    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_scores, 'bo-')
    plt.title('Silhouette Score by Number of Clusters', fontsize=16)
    plt.xlabel('Number of Clusters (k)', fontsize=14)
    plt.ylabel('Silhouette Score', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('silhouette_scores.png')
    print("Saved silhouette scores plot.")
    
    # Choose the best k
    best_k = k_range[np.argmax(silhouette_scores)]
    print(f"\nBest number of clusters based on silhouette score: {best_k}")
    
    # Apply KMeans with the best k
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to the original data
    geo_df['Cluster'] = cluster_labels
    
    # Visualize the clusters on a regular plot
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(geo_df['LONGITUDE'], geo_df['LATITUDE'], 
                          c=geo_df['Cluster'], cmap='viridis', 
                          s=10, alpha=0.6)
    
    # Plot cluster centers
    centers = scaler.inverse_transform(kmeans.cluster_centers_)
    plt.scatter(centers[:, 1], centers[:, 0], c='red', s=200, alpha=0.8, 
                marker='X', edgecolors='black', label='Cluster Centers')
    
    plt.title(f'Crime Hotspots - KMeans Clustering (k={best_k})', fontsize=16)
    plt.xlabel('Longitude', fontsize=14)
    plt.ylabel('Latitude', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig('crime_clusters.png')
    print("Saved crime clusters plot.")
    
    # Create an interactive map with clusters
    cluster_map = folium.Map(
        location=[geo_df['LATITUDE'].mean(), geo_df['LONGITUDE'].mean()],
        zoom_start=11
    )
    
    #colors for clusters
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'darkblue', 
              'darkgreen', 'cadetblue', 'lightred', 'beige', 'pink']
    
    # Create a marker cluster for each cluster
    for cluster_num in range(best_k):
        cluster_data = geo_df[geo_df['Cluster'] == cluster_num]
        marker_cluster = MarkerCluster(
            name=f'Cluster {cluster_num}',
            overlay=True
        ).add_to(cluster_map)
        
        # Add points to this cluster
        for idx, row in cluster_data.sample(min(1000, len(cluster_data))).iterrows():
            folium.CircleMarker(
                location=[row['LATITUDE'], row['LONGITUDE']],
                radius=3,
                popup=f"Crime: {row['PRIMARY_TYPE']}<br>Location: {row['LOCATION_DESCRIPTION'] if 'LOCATION_DESCRIPTION' in row else 'N/A'}<br>Cluster: {cluster_num}",
                color=colors[cluster_num % len(colors)],
                fill=True,
                fill_opacity=0.7
            ).add_to(marker_cluster)
        
        # Add cluster center
        folium.Marker(
            location=[centers[cluster_num, 0], centers[cluster_num, 1]],
            icon=folium.Icon(color=colors[cluster_num % len(colors)], icon='star'),
            popup=f'Cluster {cluster_num} Center'
        ).add_to(cluster_map)
    
    # Add layer control
    folium.LayerControl().add_to(cluster_map)
    
    # Save the interactive map
    cluster_map.save('crime_cluster_map.html')
    print("Saved interactive crime cluster map.")
    
    # Analyze crime types within each cluster
    print("\nAnalyzing crime types within each cluster...")
    cluster_crime_counts = geo_df.groupby(['Cluster', 'PRIMARY_TYPE']).size().reset_index(name='Count')
    
    # For each cluster, find the top crimes
    for cluster in range(best_k):
        cluster_crimes = cluster_crime_counts[cluster_crime_counts['Cluster'] == cluster]
        top_crimes = cluster_crimes.sort_values('Count', ascending=False).head(5)
        
        print(f"\nCluster {cluster} top crimes:")
        for _, row in top_crimes.iterrows():
            print(f"  {row['PRIMARY_TYPE']}: {row['Count']} incidents")
    
    # Calculate arrest rates by cluster
    arrest_by_cluster = geo_df.groupby('Cluster')['ARREST'].agg(['mean', 'count']).reset_index()
    arrest_by_cluster.columns = ['Cluster', 'Arrest Rate', 'Count']
    
    print("\nArrest rates by cluster:")
    for _, row in arrest_by_cluster.iterrows():
        print(f"  Cluster {int(row['Cluster'])}: {row['Arrest Rate']:.2f} ({row['Count']} incidents)")
    
    # Plot arrest rates by cluster
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Cluster', y='Arrest Rate', data=arrest_by_cluster)
    plt.title('Arrest Rate by Crime Cluster', fontsize=16)
    plt.xlabel('Cluster', fontsize=14)
    plt.ylabel('Arrest Rate', fontsize=14)
    plt.ylim(0, 1)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('arrest_rate_by_cluster.png')
    print("Saved arrest rate by cluster plot.")
    
    # Function to predict cluster for new location
    def predict_crime_cluster(latitude, longitude):
        """
        Predict which crime cluster a new location would belong to
        
        Parameters:
        latitude (float): Latitude of the location
        longitude (float): Longitude of the location
        
        Returns:
        int: Predicted cluster number
        """
        # Create a feature array
        location_features = np.array([[latitude, longitude]])
        
        # Scale the features
        scaled_features = scaler.transform(location_features)
        
        # Predict the cluster
        cluster = kmeans.predict(scaled_features)[0]
        
        return cluster
    
    # Example of using the cluster prediction function
    print("\nExample: Predicting cluster for a sample location")
    
    # Use a location close to one of the cluster centers
    sample_lat, sample_long = centers[0, 0], centers[0, 1]
    print(f"Sample location: Latitude={sample_lat:.6f}, Longitude={sample_long:.6f}")
    
    predicted_cluster = predict_crime_cluster(sample_lat, sample_long)
    print(f"Predicted cluster: {predicted_cluster}")
    
    # Get top crimes in this cluster
    top_crimes_in_cluster = cluster_crime_counts[cluster_crime_counts['Cluster'] == predicted_cluster]
    top_crimes_in_cluster = top_crimes_in_cluster.sort_values('Count', ascending=False).head(3)
    
    print("Top 3 crimes in this cluster:")
    for _, row in top_crimes_in_cluster.iterrows():
        print(f"  {row['PRIMARY_TYPE']}: {row['Count']} incidents")
    
else:
    print("Cannot perform clustering - geographic coordinates are missing.")

print("\nAnalysis completed successfully!")

# function that combines both models for comprehensive crime analysis
def analyze_new_crime(crime_data):
    """
    Analyze a new crime incident using both models
    
    Parameters:
    crime_data (dict): Dictionary containing crime details including:
                      - PRIMARY_TYPE
                      - LOCATION_DESCRIPTION (if available)
                      - LATITUDE
                      - LONGITUDE
                      - Other features used in the arrest prediction model
    
    Returns:
    dict: Analysis results including arrest prediction and cluster analysis
    """
    results = {}
    
    # Check if Random Forest model is available
    if 'predict_arrest' in locals():
        # Predict arrest
        arrest_result, arrest_prob = predict_arrest(crime_data)
        results['arrest_prediction'] = {
            'will_be_arrested': bool(arrest_result),
            'arrest_probability': float(arrest_prob)
        }
    
    # Check if K-Means clustering model is available
    if 'predict_crime_cluster' in locals() and 'LATITUDE' in crime_data and 'LONGITUDE' in crime_data:
        # Predict cluster
        cluster = predict_crime_cluster(crime_data['LATITUDE'], crime_data['LONGITUDE'])
        # Predict cluster
        cluster = predict_crime_cluster(crime_data['LATITUDE'], crime_data['LONGITUDE'])
        
        # Get relevant information about this cluster
        cluster_info = {}
        cluster_info['cluster_number'] = int(cluster)
        
        # Get top crimes in this cluster
        if 'cluster_crime_counts' in locals():
            top_crimes = cluster_crime_counts[cluster_crime_counts['Cluster'] == cluster]
            top_crimes = top_crimes.sort_values('Count', ascending=False).head(3)
            cluster_info['top_crimes'] = [{
                'type': row['PRIMARY_TYPE'],
                'count': int(row['Count'])
            } for _, row in top_crimes.iterrows()]
        
        # Get arrest rate in this cluster
        if 'arrest_by_cluster' in locals():
            cluster_arrest_info = arrest_by_cluster[arrest_by_cluster['Cluster'] == cluster]
            if not cluster_arrest_info.empty:
                cluster_info['arrest_rate'] = float(cluster_arrest_info.iloc[0]['Arrest Rate'])
                cluster_info['incident_count'] = int(cluster_arrest_info.iloc[0]['Count'])
        
        results['cluster_analysis'] = cluster_info
    
    return results

# Example usage of the combined analysis function
print("\nExample of combined crime analysis:")
example_crime = {
    'PRIMARY_TYPE': 'THEFT',
    'LOCATION_DESCRIPTION': 'STREET',
    'HOUR': 14,
    'DAY_OF_WEEK': 2,  # Tuesday
    'MONTH': 6,
    'DISTRICT': 7,
    'WARD': 32,
    'COMMUNITY_AREA_NUMBER': 5,
    'LATITUDE': sample_lat,
    'LONGITUDE': sample_long
}

print("Crime details:")
for key, value in example_crime.items():
    print(f"  {key}: {value}")

try:
    analysis_result = analyze_new_crime(example_crime)
    print("\nAnalysis results:")
    
    if 'arrest_prediction' in analysis_result:
        arrest_pred = analysis_result['arrest_prediction']
        print(f"Arrest prediction: {'Yes' if arrest_pred['will_be_arrested'] else 'No'}")
        print(f"Arrest probability: {arrest_pred['arrest_probability']:.2f}")
    
    if 'cluster_analysis' in analysis_result:
        cluster_info = analysis_result['cluster_analysis']
        print(f"Crime cluster: {cluster_info['cluster_number']}")
        
        if 'top_crimes' in cluster_info:
            print("Top crimes in this cluster:")
            for crime in cluster_info['top_crimes']:
                print(f"  {crime['type']}: {crime['count']} incidents")
        
        if 'arrest_rate' in cluster_info:
            print(f"Cluster arrest rate: {cluster_info['arrest_rate']:.2f}")
            print(f"Total incidents in cluster: {cluster_info['incident_count']}")
except Exception as e:
    print(f"Error during analysis: {e}")

print("\nChicago Crime Analysis Complete!")