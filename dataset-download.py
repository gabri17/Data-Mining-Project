import kagglehub

# Download latest version
path = kagglehub.dataset_download("wafaaelhusseini/capital-weather-data-1995-2024")

print("Path to dataset files:", path)