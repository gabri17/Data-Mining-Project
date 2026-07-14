import pandas as pd
import json

df = pd.read_parquet("history.parquet")

# Remove all rows for Mogadishu
df = df[df['capital'] != 'Mogadishu']

# Add season column based on month
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Autumn'

def get_decade(year):
    if 1995 <= year <= 2004:
        return '1995-2004'
    elif 2005 <= year <= 2014:
        return '2005-2014'
    elif 2015 <= year <= 2024:
        return '2015-2024'
    else:
        return 'Unknown'
    
def get_group(year):
    if year <= 2009:
        return 1
    else:
        return 2



df['date'] = pd.to_datetime(df['date'])

df['season'] = df['date'].dt.month.apply(get_season)
df['temp_variation'] = df['temp_max_c'] - df['temp_min_c']
df['year'] = df['date'].dt.year
df['decade'] = df['year'].apply(get_decade)
df['group'] = df['year'].apply(get_group)

#Mapping capital <-> countries
capital_to_country = df[['capital', 'country']].drop_duplicates().set_index('capital')['country'].to_dict()

columns_to_keep = ['temp_mean_c_approx', 'rain_mm', 'snow_mm', 'windspeed_10m_max_kmh', 'temp_variation', 'sunshine_duration_s', 'daylight_duration_s', 'capital', 'year', 'season', 'date', 'decade', 'group']
df = df[columns_to_keep]

#drop wind outliers

df.to_parquet("cleaned_history.parquet", index=False)

with open("capital_to_country.json", "w") as f:
    json.dump(capital_to_country, f)
    
print("Files successfully saved!")