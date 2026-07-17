features = ['mean_temp', 'std_temp', 'rain_total', 'days_rain', 'snow_total', 'days_snow', 'mean_daylight', 'mean_sunshine']
X = df_clusters[features].copy()

#Log transform for asymmetric
X["rain_total"] = np.log1p(X["rain_total"])
X["snow_total"] = np.log1p(X["snow_total"])

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

"""

# Remapping cluster idx to be align
cluster_mapping = {
    0: 4, 
    1: 2, 
    2: 3, 
    3: 0, 
    4: 1
}

df_clusters['cluster'] = df_clusters['cluster'].map(cluster_mapping)


"""
plot_something(inertia, K_range, "Inertia") #the lower the better - but see elbow
plot_something(sil_scores, K_range, "Silhouette scores") #the higher the better
plot_something(dbi_scores, K_range, "DBI scores") #the lower better (0 is the top)
plot_something(ch_scores, K_range, "CH scores") #the higher the better
plot_something(gaps, K_range, "Gap statistic") #find plateau
plot_something(bic_vals, K_range, "BIC statistic") #the lower the better - but see elbow
plot_something(aic_vals, K_range, "AIC statistic") #the lower the better - but see elbow


