# Chicago Crime Data Analysis  

This project provides insights into the **Chicago Crime Dataset** using Python and machine learning models built with **scikit-learn**.  

## Models Implemented  
1. **Random Forest Classifier**  
   - Predicts the likelihood of an arrest for a given crime.  
   - Considers factors such as the crime type and the area where it occurred.  

2. **K-Means Clustering**  
   - Uses **Silhouette Score** to determine optimal clustering.  
   - Helps identify high-crime zones where police can strengthen patrolling and surveillance.  

## Visualization  
- **Folium** and **Matplotlib** are used to generate interactive crime maps and plots.  
- Running the code will also generate **HTML reports** with visualizations.  

## Project Files  
- `P1.py` → A simplified version of the analysis for quick insights.  
- `P2.py` → A detailed version with extended exploration, advanced plots, and deeper insights.  

## Installation  

Before running the code, install the required libraries:  

```bash
pip install pandas numpy seaborn scikit-learn folium scipy
