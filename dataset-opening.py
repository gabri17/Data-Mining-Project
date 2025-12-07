import pandas as pd
from datetime import timedelta

df = pd.read_parquet("history.parquet")
print(f"We have {len(df)} in total\n")
print(f"We have these {df.columns}\n")

print(f"Datatypes of all columns:\n {df.dtypes}\n")
df['date'] = pd.to_datetime(df['date'])

num_different_cities = df['capital'].nunique()
print(f"Number of different cities: {num_different_cities}")

city_counts = df['capital'].value_counts()
print(f"Number of rows per city {city_counts}\n")

max_count = city_counts.max()
cities_not_max = city_counts[city_counts != max_count]
print(f"Maximum number of rows per city: {max_count}\n")
print(f"Cities with number of rows != from max: {len(cities_not_max)} over {num_different_cities}\n")
print(f"And it is: {cities_not_max}\n")

# Find first and last dates
first_date = df['date'].min()
last_date = df['date'].max()

print(f"First date: {first_date}")
print(f"Last date: {last_date}")
print(f"Total date: {df['date'].nunique()}")

days_difference = (last_date - first_date).days
print(f"Days from first to last date: {days_difference}")
#if days_difference = total_dates - 1 no days skipped