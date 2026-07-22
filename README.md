# Data-Mining-Project

Data analysis of the dataset: https://www.kaggle.com/datasets/wafaaelhusseini/capital-weather-data-1995-2024

Index

## Project structure
Files:
- history.parquet: dataset
- dataset-download.py: script to download dataset
- standard_pre_processing.py: script to clean and pre-process the dataset
- - capital_to_country.json -> mapping capital -> country
- - cleaned_history.parquet -> cleaned dataset
- eda.ipynb: python notebook for showing data processing, data cleaning and data exploration

- cluster.ipynb: notebook describing clustering approach to captials
- cluster_definitivo_globale_annuale.ipynb: notebook describing clustering approach to show metereological data evolution
- cluster_definitivo_globale_annuale_STAGIONALE.ipynb: notebook describing clustering approach to show metereological and seasonal data evolution

## Setup notebooks and dashboards

To setup *notebooks*, just execute at the top the cell installing the needed Python packages.

To setup *dashboards*:
1. Create a virtual environment: `python -m venv venv`
2. Activate the virtual environment:
    - Windows: `venv\Scripts\activate.bat`
    - Linux and mac: `source venv/bin/activate`
3. Install dependencies:

    `pip install --upgrade pip`

    `pip install -r requirements.txt`
4. If you want to use the virtual environment also for notebook, connect it to the notebook:

`python -m ipykernel install --user --name=climate_clustering_project --display-name "Climate Clustering Project"`

## Run dashboard locally

After having installed the needed dependencies (either directly on your machine or in the virtual environment), you can run the two dashboards following these instructions:

1. Be sure to <u>be in the SOURCE directory</u> of the project (not in the sub-directory containing *app.py*, otherwise files will not be imported properly)

2. Execute `streamlit run .\dashboard_annuale\app.py` for annual dashboard.

2. Execute `streamlit run .\dashboard_annuale_stagionale\app.py` for annual seasonal dashboard.

## Methodology

## Results