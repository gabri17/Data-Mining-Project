# Data-Mining-Project

Climate analysis and clustering of global capital cities based on historical weather data (1995-2024).

**Dataset:** [Capital Weather Data (Kaggle)](https://www.kaggle.com/datasets/wafaaelhusseini/capital-weather-data-1995-2024)

This project performs a comprehensive analysis of historical weather data for major capital cities worldwide, spanning from 1995 to 2024. The main objectives are:

- **Exploratory Data Analysis (EDA):** Understand global weather patterns, identify trends and detect outliers.
- **Clustering Analysis:** Group capital cities by their meteorological profiles using various clustering algorithms.
- **Temporal Evolution:** Track how climate profiles of cities evolve over *years* and *seasons*.
- **Interactive Dashboards:** Visualize clustering results through Streamlit-based dashboards.

---

## Table of Contents

- [Project Structure](#project-structure)
- [Methodology](#methodology)
  - [Data Preprocessing](#data-preprocessing)
  - [Exploratory Data Analysis](#exploratory-data-analysis)
  - [Clustering Approach](#clustering-approach)
  - [Clustering Variants](#clustering-variants)
- [Dashboards](#dashboards)
- [Setup](#setup)
  - [Notebooks](#notebooks)
  - [Dashboards Setup](#dashboards-setup)
- [Run Dashboard Locally](#run-dashboard-locally)
- [Results](#results)

---

## Project Structure

```
Data-Mining-Project/
├── 1_eda.ipynb                              # Exploratory Data Analysis notebook
├── 2_general_clustering.ipynb               # General clustering of capitals (all years)
├── 3_yearly_clustering.ipynb                # Year-by-year clustering evolution
├── 4_seasonal_clustering.ipynb              # Seasonal clustering evolution
│
├── dataset-download.py                      # Script to download dataset from Kaggle
├── standard_pre_processing.py               # Data cleaning and preprocessing script
│
├── history.parquet                          # Raw dataset
│
├── cleaned_history.parquet                  # Cleaned and preprocessed dataset
├── capital_to_country.json                  # Mapping: capital -> country
├── capital_to_lat.json                      # Mapping: capital -> latitude
├── capital_to_lon.json                      # Mapping: capital -> longitude
│
├── requirements.txt                         # Python dependencies
│
├── dashboard_annuale/                       # Annual clustering dashboard
│   ├── app.py
│   ├── csv_files/
│   └── requirements.txt
├── dashboard_annuale_stagionale/            # Seasonal clustering dashboard
│   ├── app.py
│   ├── csv_files/
│   └── requirements.txt
└── README.md
```

---

## Methodology

### Data Preprocessing

1. **Data Source:** Daily weather records for 160+ capital cities from 1995 to 2024.
2. **Cleaning Steps:**
   - Removed records for Mogadishu (incomplete data).
   - Converted date columns to datetime format.
   - Handled technical outliers (e.g., wind speeds > 150 km/h, apparent temperatures < -60°C) using 7-day centered rolling means.
3. **Feature Engineering:**
   - Added `season` column (Winter, Spring, Summer, Autumn).
   - Added `temp_variation` (difference between max and min temperature).
   - Added `year`, `decade` and `group` columns for temporal analysis.
4. **Final Features for clustering:** `temp_mean_c_approx`, `rain_mm`, `snow_mm`, `windspeed_10m_max_kmh`, `temp_variation`, `sunshine_duration_s`, `daylight_duration_s`.

### Exploratory Data Analysis

The EDA notebook (`1_eda.ipynb`) covers:
- Dataset structure and summary statistics.
- Temperature, precipitation, and wind distributions.
- Correlation analysis between weather variables.
- Identification of extreme weather events.
- Geospatial visualization of global temperature patterns.

### Clustering Approach

1. **Feature Selection:** PCA analysis to identify the most informative features.
2. **Feature Aggregation:** Monthly/seasonal aggregations of daily data.
3. **Data Scaling:** StandardScaler applied to normalize features.
4. **Algorithms Evaluated:**
   - **K-Means:** Partitional clustering with elbow method and silhouette analysis.
   - **DBSCAN:** Density-based clustering for discovering arbitrary-shaped clusters.
   - **Hierarchical Clustering:** Agglomerative approach with dendrogram visualization.
   - **BIRCH:** Efficient clustering for large datasets.
5. **Metrics:** Silhouette Score, Davies-Bouldin Index, Calinski-Harabasz Index.

### Clustering Variants

- **General Clustering** (`2_general_clustering.ipynb`): Clusters all capitals using aggregated data across the full time range.
- **Yearly Clustering** (`3_yearly_clustering.ipynb`): Clusters capitals year-by-year to track climate profile changes over time.
- **Seasonal Clustering** (`4_seasonal_clustering.ipynb`): Clusters season-city pairs year-by-year to analyze intra-annual variability.

---

## Dashboards

Two interactive Streamlit dashboards are provided:

| Dashboard | Description | Local deployment | Available link |
|-----------|-------------|------------------|----------------|
| **Annual** | Visualizes yearly clustering results | `streamlit run dashboard_annuale\app.py` | <a href="https://climacluster.streamlit.app">climacluster.streamlit.app</a> |
| **Seasonal** | Visualizes seasonal clustering results | `streamlit run dashboard_annuale_stagionale\app.py` | <a href="https://seasoncluster.streamlit.app">seasoncluster.streamlit.app</a> |  

---

## Setup

### Notebooks

To run the notebooks, execute the first cell in each notebook to install the required Python packages:

```bash
pip install pandas matplotlib seaborn plotly geopandas scipy pyarrow scikit-learn numpy
```

### Dashboards Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```

2. Activate the virtual environment:
   - **Windows:**
     ```bash
     venv\Scripts\activate.bat
     ```
   - **Linux/macOS:**
     ```bash
     source venv/bin/activate
     ```

3. Install dependencies:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. (Optional) Connect the virtual environment to Jupyter notebooks:
   ```bash
   python -m ipykernel install --user --name=climate_clustering_project --display-name "Climate Clustering Project"
   ```

---

## Run Dashboard Locally

After installing dependencies, run the dashboards from the **project root directory** (not from the dashboard subdirectory):

```bash
# Annual dashboard
streamlit run dashboard_annuale\app.py

# Seasonal dashboard
streamlit run dashboard_annuale_stagionale\app.py
```


---

## Results

- **EDA:** Identified global temperature gradients, seasonal patterns
 and extreme weather events across 160+ capitals.
- **Clustering:** Discovered distinct climate groups among world capitals (e.g., tropical, arid, temperate, continental).
- **Yearly Evolution:** Tracked how cities shift between climate clusters over the 30-year period, highlighting potential climate change impacts.
- **Seasonal Analysis:** Revealed intra-annual climate variability and seasonal cluster transitions.

Detailed results and visualizations can be found in the respective notebooks, in the dashboards and in the report attached.

<img width="1415" height="718" alt="clustering_map (1)" src="https://github.com/user-attachments/assets/80583c0c-6539-44ce-8160-f0b69f0aa006" />

Clustering has found **distinct** and **recognizable** climate zones.
* **Cluster 0 (Blue) Subarctic and Cold Continental**
* **Cluster 1 (Orange) Tropical Savanna and Warm Semi-Arid**
* **Cluster 2 (Green) Temperate and Mid-Latitude**
* **Cluster 3 (Red) Equatorial Rainforest and Monsoon**
* **Cluster 4 (Purple) Arid, Desert and Dry Subtropical**
