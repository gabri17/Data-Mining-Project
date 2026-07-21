import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.preprocessing import MinMaxScaler

# ==============================================================================
# 1. CONFIGURAZIONE E CARICAMENTO DATI
# ==============================================================================
st.set_page_config(page_title="Climate Cluster Dashboard - Seasonal Yearly analysis (1995-2024)", layout="wide")

@st.cache_data
def load_raw_data(file_path="dashboard_annuale_stagionale/csv_files/aggregation_per_season_anno.csv"):
    """
    Carica i dati con le medie annuali delle capitali.
    """
    try:
        aggregation_per_year = pd.read_csv(file_path)
        return aggregation_per_year
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return None

@st.cache_data
def load_data(directory_path="dashboard_annuale_stagionale/csv_files", season="Winter"):
    """
    Carica i dati di clusterizzazione per la mappa animata (filtrati per stagione).
    """
    years = list(range(1995, 2025))
    df_clusters_anni = {}
    cluster_means_anni = {}
    for year in years:
        try:
            df_year = pd.read_csv(f"{directory_path}/clusterization_{year}.csv")
            df_clusters_anni[year] = df_year[df_year['season'] == season]
            cluster_means_anni[year] = pd.read_csv(f"{directory_path}/clusterization_means_{year}.csv")
        except FileNotFoundError:
            pass 

    return df_clusters_anni, cluster_means_anni

@st.cache_data
def load_all_seasons_data(directory_path="dashboard_annuale_stagionale/csv_files"):
    """
    Carica un unico dataframe con tutte le stagioni e tutti gli anni 
    per le analisi storiche a matrice.
    """
    years = list(range(1995, 2025))
    records = []
    for year in years:
        try:
            df_year = pd.read_csv(f"{directory_path}/clusterization_{year}.csv")
            df_year['year'] = year
            records.append(df_year)
        except FileNotFoundError:
            pass 
            
    if records:
        return pd.concat(records, ignore_index=True)
    return pd.DataFrame()

@st.cache_data
def get_colors():
    cmap = plt.get_cmap("tab10")
    hex_codes = [mcolors.to_hex(cmap(i)) for i in range(cmap.N)]
    my_colors = {i: hex_codes[i] for i in range(len(hex_codes))}
    return my_colors

# ==============================================================================
# GRAFICI E PLOT
# ==============================================================================

def plot_world_slider_streamlit_go(df, years, n_clusters=5, method="K-Means", season="Winter"):
    """
    Crea una mappa coropletica animata con slider e tasto play per Streamlit.
    """
    colors_to_use = get_colors()
    if colors_to_use is None:
        colors_to_use = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    colorscale = []
    for i in range(n_clusters):
        c = colors_to_use[i % len(colors_to_use)]
        colorscale.append([i / n_clusters, c])
        colorscale.append([(i + 1) / n_clusters, c])

    frames = []
    for year in years:
        if year not in df:
            continue
        df_year = df[year]
        
        frames.append(
            go.Frame(
                data=[
                    go.Choropleth(
                        locations=df_year["country"],
                        locationmode="country names",
                        z=df_year["cluster"],
                        zmin=0, zmax=n_clusters - 1,
                        colorscale=colorscale,
                        showscale=True,
                        colorbar=dict(title="Cluster", tickvals=list(range(n_clusters)), len=0.75),
                        marker_line_color="black", marker_line_width=0.5,
                        customdata=np.stack([df_year["capital"], df_year["country"], df_year["cluster"]], axis=-1),
                        hovertemplate=(
                            "<b>%{customdata[1]}</b><br>" 
                            "Capital: %{customdata[0]}<br>" 
                            "Cluster: <b>%{customdata[2]}</b><extra></extra>"
                        ),
                    ),
                    go.Scattergeo(
                        lon=df_year['lon'], lat=df_year['lat'],
                        text=df_year["capital"], mode="markers",
                        marker=dict(size=4, color='black', opacity=0.95, line=dict(color="white", width=0.6)),
                        hovertemplate="<b>%{text}</b><br>Cluster: %{customdata[0]}<extra></extra>",
                        customdata=np.stack([df_year["cluster"]], axis=-1),
                        showlegend=False
                    )
                ],
                name=str(year)
            )
        )

    if not frames:
        return None

    fig = go.Figure(
        data=frames[0].data,
        layout=go.Layout(
            title=dict(text=f"Climate Clusters {season} {years[0]} - {years[-1]} ({method})", x=0.5, xanchor="center", font=dict(size=20)),
            width=1400, height=750,
            geo=dict(domain=dict(x=[0, 1], y=[0.1, 1]), projection_type="natural earth", showframe=False, showcountries=True, countrycolor="black", countrywidth=1.2, showcoastlines=False, bgcolor="white"),
            updatemenus=[{
                "type": "buttons", "showactive": False, "x": 0.5, "y": 0.03, "xanchor": "center", "direction": "left",
                "buttons": [
                    {"label": "▶ Play", "method": "animate", "args": [None, {"frame": {"duration": 700, "redraw": True}, "fromcurrent": True, "transition": {"duration": 300}}]},
                    {"label": "⏸ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]}
                ]
            }],
            sliders=[{
                "y": 0.05, "x": 0.1, "len": 0.8,
                "steps": [{"method": "animate", "label": str(year), "args": [[str(year)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]} for year in years if year in df],
                "currentvalue": {"prefix": "Year: ", "font": {"size": 16}}
            }]
        ),
        frames=frames
    )
    return fig

def plot_city_cluster_matrix(full_df, city_name, colors_to_use):
    """
    Plots a Season x Year grid for a single city showing cluster trajectory.
    """
    city_df = full_df[full_df['capital'] == city_name].copy()

    if city_df.empty:
        st.warning(f"Nessun dato trovato per la città: {city_name}")
        return None

    colors_list = list(colors_to_use.values())

    # Enforce standard seasonal order
    season_order = ['Winter', 'Spring', 'Summer', 'Autumn']
    city_df['season'] = pd.Categorical(city_df['season'], categories=season_order, ordered=True)

    # Pivot table: Rows = Seasons, Columns = Years
    pivot_df = city_df.pivot(index='season', columns='year', values='cluster')
    pivot_df = pivot_df.apply(pd.to_numeric, errors='coerce')

    # Plot discrete heatmap
    fig, ax = plt.subplots(figsize=(14, 4))
    cmap = ListedColormap(colors_list)

    sns.heatmap(
        pivot_df,
        annot=True,              # Print cluster number inside the cell
        fmt=".0f",               # Formats numbers cleanly
        cmap=cmap,
        vmin=0,
        vmax=len(colors_list) - 0.01,
        linewidths=1.5,
        linecolor='white',
        cbar=False,              # Hide colorbar since colors match cluster IDs
        mask=pivot_df.isna(),    # Masks out missing data safely
        ax=ax
    )

    ax.set_title(f'Seasonal Cluster Trajectory: {city_name}', fontsize=15, pad=15)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Season', fontsize=12)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    return fig

def plot_temp_evo(cluster_means_anni, year_period, n_clusters=5):
    comparison_data = {}
    for year in year_period:
        if year in cluster_means_anni:
            comparison_data[year] = cluster_means_anni[year]['mean_temp']

    comparison_temp = pd.DataFrame(comparison_data)

    fig = plt.figure(figsize=(16, 8))
    for cluster_id in range(n_clusters):
        if cluster_id in comparison_temp.index:
            temps = comparison_temp.loc[cluster_id]
            plt.plot(year_period, temps, marker='o', label=f'Cluster {cluster_id}', linewidth=2, markersize=8)

    plt.xlabel('Anno', fontsize=12)
    plt.ylabel('Temperatura Media (°C)', fontsize=12)
    plt.title('Evoluzione Temperature Medie per Cluster negli Anni', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_precip_evo(cluster_means_anni, year_period, n_clusters=5):
    comparison_rain = {}
    for year in year_period:
        if year in cluster_means_anni:
            comparison_rain[year] = cluster_means_anni[year]['rain_total']

    comparison_rain_df = pd.DataFrame(comparison_rain)

    fig = plt.figure(figsize=(16, 8))
    for cluster_id in range(n_clusters):
        if cluster_id in comparison_rain_df.index:
            precips = comparison_rain_df.loc[cluster_id]
            plt.plot(year_period, precips, marker='o', label=f'Cluster {cluster_id}', linewidth=2, markersize=8)

    plt.xlabel('Anno', fontsize=12)
    plt.ylabel('Precipitazioni annuali (mm)', fontsize=12)
    plt.title('Evoluzione Precipitazioni Annuali per Cluster negli Anni', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_temp_evo_city(cluster_means_anni, df_clusters_anni, aggregated_data, year_period, city):
    aggregated_data_city = aggregated_data[aggregated_data["capital"] == city].copy()
    city_temps = aggregated_data_city.set_index("year")["mean_temp"]
    
    cluster_temps = {}
    cluster_ids = {}
    
    for year in year_period:
        df_year = df_clusters_anni.get(year)
        if df_year is None:
            continue
        row = df_year[df_year["capital"] == city]
        if row.empty:
            continue
        cluster_id = int(row["cluster"].iloc[0])
        cluster_ids[year] = cluster_id
        try:
            cluster_temps[year] = cluster_means_anni[year].loc[cluster_id, "mean_temp"]
        except KeyError:
            cluster_temps[year] = np.nan
    
    fig = plt.figure(figsize=(16, 8))
    plt.plot(
        year_period,
        [city_temps.get(year, np.nan) for year in year_period],
        marker="o", label=f"{city}", linewidth=2, color="#8c564b"
    )
    plt.plot(
        year_period,
        [cluster_temps.get(year, np.nan) for year in year_period],
        marker="s", label="Cluster di appartenenza", linewidth=2, color="#7f7f7f"
    )

    colors = get_colors()
    for year in year_period:
        curr_cluster = cluster_ids.get(year)
        temp_val = cluster_temps.get(year, np.nan)
        if not np.isnan(temp_val):
            plt.annotate(
                f"C{curr_cluster}", xy=(year, temp_val), xytext=(0, 10),
                textcoords="offset points", ha="center", fontsize=10,
                color=colors[curr_cluster], fontweight="bold",
            )
    
    plt.xlabel("Anno", fontsize=12)
    plt.ylabel("Temperatura Media (°C)", fontsize=12)
    plt.title(f"Evoluzione Temperature — {city}", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_precip_evo_city(cluster_means_anni, df_clusters_anni, aggregated_data, year_period, city):
    aggregated_data_city = aggregated_data[aggregated_data["capital"] == city].copy()
    city_precips = aggregated_data_city.set_index("year")["rain_total"]
    
    cluster_precips = {}
    cluster_ids = {}
    
    for year in year_period:
        df_year = df_clusters_anni.get(year)
        if df_year is None:
            continue
        row = df_year[df_year["capital"] == city]
        if row.empty:
            continue
        cluster_id = int(row["cluster"].iloc[0])
        cluster_ids[year] = cluster_id
        try:
            cluster_precips[year] = cluster_means_anni[year].loc[cluster_id, "rain_total"]
        except KeyError:
            cluster_precips[year] = np.nan
    
    fig = plt.figure(figsize=(16, 8))
    plt.plot(
        year_period,
        [city_precips.get(year, np.nan) for year in year_period],
        marker="o", label=f"{city}", linewidth=2, color="#8c564b"
    )
    plt.plot(
        year_period,
        [cluster_precips.get(year, np.nan) for year in year_period],
        marker="s", label="Cluster di appartenenza", linewidth=2, color="#7f7f7f"
    )

    colors = get_colors()
    for year in year_period:
        curr_cluster = cluster_ids.get(year)
        precip_val = cluster_precips.get(year, np.nan)
        if not np.isnan(precip_val):
            plt.annotate(
                f"C{curr_cluster}", xy=(year, precip_val), xytext=(0, 10),
                textcoords="offset points", ha="center", fontsize=10,
                color=colors[curr_cluster], fontweight="bold",
            )
    
    plt.xlabel("Anno", fontsize=12)
    plt.ylabel("Precipitazioni totali (mm)", fontsize=12)
    plt.title(f"Evoluzione precipitazioni — {city}", fontsize=14, fontweight="bold")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

# ==============================================================================
# MAIN APP NAVIGATION
# ==============================================================================
aggregation_per_year = load_raw_data()

st.sidebar.title("Menu")
page = st.sidebar.radio("Go to:", ["Annual Map", "City Clustering Assignments", "Capital Analysis (Radar)", "Time evolution"])
st.sidebar.markdown("---")

if aggregation_per_year is not None:
    years = sorted(aggregation_per_year['year'].unique())
    seasons = ["Winter", "Spring", "Summer", "Autumn"]
    
    # --------------------------------------------------------------------------
    # PAGINA 1: MAPPA ANNUALE
    # --------------------------------------------------------------------------
    if page == "Annual Map":
        st.title("🗺️ Seasonal Global Climate Evolution - Annual Map")
        st.sidebar.info("**Note:** Clustering applies to the *capitals*. The map colors the state directly for easier readability.")

        selected_season = st.selectbox("Seleziona la stagione", seasons, key="season_selector")
        df_clusters_anni, cluster_means_anni = load_data(season=selected_season)
        
        st.info("Press 'Play' to see how climate clusters change over time.")
        
        with st.spinner("Preparazione animazione..."):
            fig_anim = plot_world_slider_streamlit_go(
                df=df_clusters_anni, 
                years=years,
                n_clusters=5,
                season=selected_season
            )
            
            if fig_anim:
                st.plotly_chart(fig_anim, width='stretch', use_container_width=True)
            else:
                st.warning("Nessun dato disponibile per la visualizzazione della mappa.")
        
        st.markdown("---")
        st.markdown("### 📊 Proiezione Dati sul Dataframe")
        
        col_stats_1, col_stats_2 = st.columns([1, 1])
        with col_stats_1:
            available_years = sorted(list(df_clusters_anni.keys()))
            if available_years:
                stats_year = st.select_slider(
                    "Seleziona l'anno per proiettare i dati",
                    options=available_years,
                    value=available_years[-1],
                    key="stats_year_slider"
                )
        with col_stats_2:
            stats_view = st.radio(
                "Visualizzazione Dataframe",
                ["Centroidi Cluster", "Distribuzione Capitali"],
                horizontal=True,
                key="stats_view_radio"
            )

        if available_years and stats_year in cluster_means_anni and stats_year in df_clusters_anni:
            if stats_view == "Centroidi Cluster":
                st.dataframe(cluster_means_anni[stats_year], use_container_width=True)
            else:
                df_year = df_clusters_anni[stats_year].copy()
                distribuzione = (
                    df_year.groupby("cluster")
                    .agg(capitali=("capital", "count"))
                    .reset_index()
                    .rename(columns={"cluster": "Cluster"})
                )
                distribuzione["% Capitali"] = ((distribuzione["capitali"] / distribuzione["capitali"].sum()) * 100).round(2)
                st.dataframe(distribuzione, use_container_width=True)

    # --------------------------------------------------------------------------
    # PAGINA 2: TRAIETTORIA CITTÀ (HEATMAP)
    # --------------------------------------------------------------------------
    elif page == "City Clustering Assignments":
        st.title("🏙️ Traiettoria Climatica Stagionale")
        st.markdown("Osserva come i cluster climatici di una specifica capitale evolvono attraverso le stagioni nel corso degli anni.")
        
        # Load the combined dataframe once
        full_df = load_all_seasons_data()
        my_colors = get_colors()

        if not full_df.empty:
            # Grab unique cities for the dropdown
            cities = sorted(full_df['capital'].dropna().unique().tolist())
            
            selected_city = st.selectbox(
                "Seleziona la Capitale:",
                options=cities,
                index=0
            )

            with st.spinner(f"Generazione matrice per {selected_city}..."):
                fig_heatmap = plot_city_cluster_matrix(full_df, selected_city, my_colors)
                if fig_heatmap:
                    st.pyplot(fig_heatmap, use_container_width=True)
        else:
            st.warning("Dati di clusterizzazione non trovati. Verifica la cartella csv_files.")
    
    # --------------------------------------------------------------------------
    # PAGINA 3: RADAR PLOT E CONFRONTO CAPITALI
    # --------------------------------------------------------------------------
    elif page == "Capital Analysis (Radar)":
        from sklearn.preprocessing import MinMaxScaler
        
        st.title("🕸️ Detailed Analysis (Radar Plot)")
        
        col1, col2 = st.columns(2)
        capitals = sorted(aggregation_per_year['capital'].dropna().unique())
        
        # Selezione 1
        with col1:
            st.subheader("Selection 1")
            city1 = st.selectbox("Capitale 1", capitals, key="c1")
            year1 = st.selectbox("Anno 1", years, key="y1", index=len(years)-1)
            season1 = st.selectbox("Stagione 1", seasons, key="s1")
            
        # Selezione 2
        with col2:
            st.subheader("Selection 2")
            city2 = st.selectbox("Capitale 2", capitals, key="c2", index=min(1, len(capitals)-1))
            year2 = st.selectbox("Anno 2", years, key="y2", index=len(years)-1)
            season2 = st.selectbox("Stagione 2", seasons, key="s2")

        # Caricamento dinamico dei dati di cluster basato sulla stagione selezionata
        df_clusters_s1, cluster_means_s1 = load_data(season=season1)
        df_clusters_s2, cluster_means_s2 = load_data(season=season2)

        try:
            # Recupero dati
            data1 = aggregation_per_year[
                (aggregation_per_year['capital'] == city1) & 
                (aggregation_per_year['year'] == year1) & 
                (aggregation_per_year['season'] == season1)
            ].iloc[0]
            
            data2 = aggregation_per_year[
                (aggregation_per_year['capital'] == city2) & 
                (aggregation_per_year['year'] == year2) & 
                (aggregation_per_year['season'] == season2)
            ].iloc[0]
            
            # Recupero centroidi dei cluster di appartenenza
            cluster_assignment1 = df_clusters_s1[year1]
            cluster_centroids1 = cluster_means_s1[year1]
            cluster1_idx = cluster_assignment1[cluster_assignment1['capital'] == city1]['cluster'].iloc[0]
            centroid1 = cluster_centroids1.loc[cluster1_idx]
            
            cluster_assignment2 = df_clusters_s2[year2]
            cluster_centroids2 = cluster_means_s2[year2]
            cluster2_idx = cluster_assignment2[cluster_assignment2['capital'] == city2]['cluster'].iloc[0]
            centroid2 = cluster_centroids2.loc[cluster2_idx]

            to_drop = ['capital', 'year', 'season']
            
            # Rimuovi eventuali altre colonne testuali/inutili se presenti, ad es. 'country'
            for col in to_drop:
                if col in data1.index:
                    data1 = data1.drop(col)
                if col in data2.index:
                    data2 = data2.drop(col)

            comparison_df = pd.DataFrame(
                [data1, data2],
                index=[f"{city1} — {year1} ({season1}) — C{cluster1_idx}", f"{city2} — {year2} ({season2}) — C{cluster2_idx}"]
            )
            st.dataframe(comparison_df, use_container_width=True)

            comparison_df_cluster = pd.DataFrame(
                [centroid1, centroid2],
                index=[f"Cluster {cluster1_idx} di {city1} — {year1}", f"Cluster {cluster2_idx} di {city2} — {year2}"]
            )
            st.dataframe(comparison_df_cluster, use_container_width=True)

            features_to_use = ['mean_temp', 'std_temp', 'rain_total', 'days_rain', 'snow_total', 'mean_sunshine']
            
            # --- PREPARAZIONE RADAR (NORMALIZZAZIONE) ---
            scaler_radar = MinMaxScaler()

            global_scaled = aggregation_per_year[features_to_use].copy()
            for col in ['rain_total', 'snow_total']:
                global_scaled[col] = np.log1p(global_scaled[col].clip(lower=0))
            scaler_radar.fit(global_scaled)

            def prepare_vector(series_like):
                vec = series_like[features_to_use].astype(float).copy()
                vec['rain_total'] = np.log1p(max(vec['rain_total'], 0))
                vec['snow_total'] = np.log1p(max(vec['snow_total'], 0))
                vec_df = pd.DataFrame([vec.values], columns=features_to_use)
                return scaler_radar.transform(vec_df)[0]
            
            r_city1 = prepare_vector(data1)
            r_city2 = prepare_vector(data2)
            r_centroid1 = prepare_vector(centroid1)
            r_centroid2 = prepare_vector(centroid2)

            def close_radar(values):
                return np.append(values, values[0]), features_to_use + [features_to_use[0]]

            r_city1_closed, theta_closed = close_radar(r_city1)
            r_centroid1_closed, _ = close_radar(r_centroid1)
            r_city2_closed, _ = close_radar(r_city2)
            r_centroid2_closed, _ = close_radar(r_centroid2)
            
            fig_city1 = go.Figure()
            fig_city1.add_trace(go.Scatterpolar(
                r=r_city1_closed,
                theta=theta_closed,
                fill='toself',
                name=f'{city1} - Cluster {cluster1_idx}',
                line_color='blue'
            ))
            fig_city1.add_trace(go.Scatterpolar(
                r=r_centroid1_closed,
                theta=theta_closed,
                name=f'Media Cluster {cluster1_idx}',
                line=dict(color='blue', dash='dot'),
                opacity=0.5
            ))
            fig_city1.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title=f"{city1} ({season1} {year1}) vs C{cluster1_idx}"
            )

            fig_city2 = go.Figure()
            fig_city2.add_trace(go.Scatterpolar(
                r=r_city2_closed,
                theta=theta_closed,
                fill='toself',
                name=f'{city2} - Cluster {cluster2_idx}',
                line_color='red'
            ))
            fig_city2.add_trace(go.Scatterpolar(
                r=r_centroid2_closed,
                theta=theta_closed,
                name=f'Media Cluster {cluster2_idx}',
                line=dict(color='red', dash='dot'),
                opacity=0.5
            ))
            fig_city2.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title=f"{city2} ({season2} {year2}) vs C{cluster2_idx}"
            )

            col_rad1, col_rad2 = st.columns(2)
            with col_rad1:
                st.plotly_chart(fig_city1, use_container_width=True)
            with col_rad2:
                st.plotly_chart(fig_city2, use_container_width=True)
            
            st.info("ℹ️ The values in the plots are normalized (0 = global minimum, 1 = global maximum) to allow for comparison between different units of measurement (mm, °C, days).")
            
        except IndexError:
            st.error("Dati non trovati per la combinazione Capitale/Anno/Stagione selezionata. Prova a cambiare i parametri.")
    
    # --------------------------------------------------------------------------
    # PAGINA 4: EVOLUZIONE TEMPORALE
    # --------------------------------------------------------------------------
    elif page == "Time evolution":
        st.title("🗺️ Temperature and precipitation trends")
        
        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            selected_season = st.selectbox("Seleziona la stagione", seasons, key="evo_season")
        with col_sel2:
            capitals = sorted(aggregation_per_year['capital'].dropna().unique())
            selected_city = st.selectbox("Capitale", capitals, key="evo_city")

        # Carica dati dinamici basati sulla stagione scelta
        df_clusters_evo, cluster_means_evo = load_data(season=selected_season)
        
        # Filtra i dati grezzi storici per la stagione scelta
        aggregation_filtered = aggregation_per_year[aggregation_per_year["season"] == selected_season]
        
        st.markdown(f"### 📈 General performance of the clusters")
        fig1 = plot_temp_evo(cluster_means_evo, years)
        fig2 = plot_precip_evo(cluster_means_evo, years)
        
        st.pyplot(fig1, use_container_width=True)
        st.pyplot(fig2, use_container_width=True)

        st.markdown(f"### 📊 Specific Evolution: {selected_city} ({selected_season})")
        
        try:
            fig3 = plot_temp_evo_city(cluster_means_evo, df_clusters_evo, aggregation_filtered, years, selected_city)
            st.pyplot(fig3, use_container_width=True)
            
            fig4 = plot_precip_evo_city(cluster_means_evo, df_clusters_evo, aggregation_filtered, years, selected_city)
            st.pyplot(fig4, use_container_width=True)
        except Exception as e:
            st.error(f"Errore nella generazione dei grafici per la città selezionata: Dati insufficienti o mancanti.")