import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import geopandas as gpd
import plotly.graph_objects as go
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# ==============================================================================
# 1. CONFIGURAZIONE E CARICAMENTO DATI
# ==============================================================================
st.set_page_config(page_title="Climate Cluster Dashboard - Yearly analysis (1995-2024)", layout="wide")

@st.cache_data
def load_raw_data(file_path="./csv_files/aggregation_per_anno.csv"):
    """
    Carica i dati con le medie annuali delle capitali.
    """
    aggregation_per_year = pd.read_csv(file_path)

    return aggregation_per_year


@st.cache_data
def load_data(directory_path="./csv_files"):
    """
    Carica i dati di clusterizzazione e dei centroidi.
    """
    years = list(range(1995, 2025))
    df_clusters_anni = {}
    cluster_means_anni = {}
    for year in years:
        df_clusters_anni[year] = pd.read_csv(directory_path + f"/clusterization_{year}.csv")
        cluster_means_anni[year] = pd.read_csv(directory_path + f"/clusterization_means_{year}.csv")

    return df_clusters_anni, cluster_means_anni

@st.cache_data
def get_colors():
    cmap = plt.get_cmap("tab10")
    hex_codes = [mcolors.to_hex(cmap(i)) for i in range(cmap.N)]

    my_colors = {i: hex_codes[i] for i in range(len(hex_codes))}
    return my_colors


def plot_world_slider_streamlit_go(df, years=list(range(1995,2025)), n_clusters=5, method="K-Means"):
    """
    Crea una mappa coropletica animata con slider e tasto play per Streamlit.
    """

    colors_to_use=get_colors()
    
    # Gestione colori di default se non forniti
    if colors_to_use is None:
        colors_to_use = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # Costruzione della Colorscale discreta (per avere colori netti tra cluster)
    colorscale = []
    for i in range(n_clusters):
        c = colors_to_use[i % len(colors_to_use)] # % len per evitare index out of range
        colorscale.append([i / n_clusters, c])
        colorscale.append([(i + 1) / n_clusters, c])

    # --- CREAZIONE FRAMES (Uno per anno) ---
    frames = []
    for year in years:
        df_year = df[year]
        
        frames.append(
            go.Frame(
                data=[
                    go.Choropleth(
                        locations=df_year["country"],
                        locationmode="country names",
                        z=df_year["cluster"],
                        zmin=0,
                        zmax=n_clusters - 1,
                        colorscale=colorscale,
                        showscale=True,
                        colorbar=dict(
                            title="Cluster",
                            tickvals=list(range(n_clusters)),
                            len=0.75
                        ),
                        marker_line_color="black",
                        marker_line_width=0.5,
                        # Custom data per l'hover
                        customdata=np.stack(
                            [df_year["capital"], df_year["country"], df_year["cluster"]],
                            axis=-1
                        ),
                        hovertemplate=(
                            "<b>%{customdata[1]}</b><br>"  # Nazione
                            "Capital: %{customdata[0]}<br>" # Capitale
                            "Cluster: <b>%{customdata[2]}</b>" # Cluster
                            "<extra></extra>"
                        ),
                    ),
                    go.Scattergeo(
                        lon=df_year['lon'],
                        lat=df_year['lat'],
                        text=df_year["capital"],
                        mode="markers",
                        marker=dict(
                            size=4,
                            color='black',
                            opacity=0.95,
                            line=dict(color="white", width=0.6)
                        ),
                        hovertemplate="<b>%{text}</b><br>Cluster: %{customdata[0]}<extra></extra>",
                        customdata=np.stack([df_year["cluster"]], axis=-1),
                        showlegend=False
                    )
                ],
                name=str(year)
            )
        )

    # --- CREAZIONE FIGURA INIZIALE (Primo frame) ---
    # Usiamo i dati del primo anno per il layout iniziale

    # --- LAYOUT E CONTROLLI ---
    fig = go.Figure(
        data=frames[0].data,
        layout=go.Layout(
            title=dict(
                text=f"Climate Clusters {years[0]} - {years[-1]} ({method})",
                x=0.5,
                xanchor="center",
                font=dict(size=20)
            ),
            width=1400,
            height=750,
            geo=dict(
                domain=dict(x=[0, 1], y=[0.1, 1]),
                projection_type="natural earth",
                showframe=False,
                showcountries=True,
                countrycolor="black",
                countrywidth=1.2,
                showcoastlines=False,
                bgcolor="white"
            ),
            
            updatemenus=[{
                "type": "buttons",
                "showactive": False,
                "x": 0.5,
                "y": 0.03,
                "xanchor": "center",
                "direction": "left",
                "buttons": [
                    {
                        "label": "▶ Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 700, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 300}
                            }
                        ]
                    },
                    {
                        "label": "⏸ Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate"
                            }
                        ]
                    }
                ]
            }],

            sliders=[{
                "y": 0.05,
                "x": 0.1,
                "len": 0.8,
                "steps": [
                    {
                        "method": "animate",
                        "label": str(year),
                        "args": [
                            [str(year)],
                            {
                                "frame": {"duration": 0, "redraw": True},
                                "mode": "immediate"
                            }
                        ]
                    }
                    for year in years
                ],
                "currentvalue": {
                    "prefix": "Year: ",
                    "font": {"size": 16}
                }
            }]
        ),
        frames=frames
    )

    return fig


def plot_comparison_maps(df, year_start, year_end, n_clusters=5):
    """
    Crea due mappe statiche affiancate per confrontare due anni specifici.
    - Colori: indicano il cluster.
    - Intensità (Opacità): 
        - ALTA se il paese ha cambiato cluster tra i due anni.
        - BASSA se il paese è rimasto nello stesso cluster.
    """
    
    colors_to_use = get_colors()
    
    # Costruzione della Colorscale discreta (identica alla tua funzione precedente)
    colorscale = []
    for i in range(n_clusters):
        c = colors_to_use[i % len(colors_to_use)]
        colorscale.append([i / n_clusters, c])
        colorscale.append([(i + 1) / n_clusters, c])

    # 2. Preparazione Dati
    df1 = df[year_start].copy()
    df2 = df[year_end].copy()
    
    # Uniamo i dataframe per capire chi è cambiato
    # Assumiamo che 'country' sia la chiave univoca
    merged = df1[['country', 'cluster']].merge(
        df2[['country', 'cluster']], 
        on='country', 
        suffixes=('_start', '_end')
    )
    
    # Identifichiamo i paesi che hanno cambiato cluster
    # True se cambiato, False se uguale
    changed_mask = merged['cluster_start'] != merged['cluster_end']
    changed_countries = merged.loc[changed_mask, 'country'].values
    
    # Funzione helper per dividere il df in "stabili" e "cambiati"
    def split_stable_changed(df_year):
        is_changed = df_year['country'].isin(changed_countries)
        return df_year[~is_changed], df_year[is_changed]

    df1_stable, df1_changed = split_stable_changed(df1)
    df2_stable, df2_changed = split_stable_changed(df2)

    # 3. Creazione Subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'geo'}, {'type': 'geo'}]],
        subplot_titles=(f"Anno {year_start}", f"Anno {year_end}"),
        horizontal_spacing=0.05
    )

    # Funzione helper per aggiungere le tracce
    def add_traces(df_stable, df_changed, col_index, show_colorbar=False):
        
        # TRACCIA 1: STABILI (Opacità ridotta)
        if not df_stable.empty:
            fig.add_trace(
                go.Choropleth(
                    locations=df_stable["country"],
                    locationmode="country names",
                    z=df_stable["cluster"],
                    colorscale=colorscale,
                    zmin=0, zmax=n_clusters - 1,
                    showscale=False, # Niente legenda per lo sfondo
                    marker_opacity=0.2, # Bassa opacità
                    marker_line_color='gainsboro',
                    marker_line_width=0.5,
                    name="Stable",
                    hoverinfo="skip" # Rimuoviamo hover inutile sugli sfondi
                ),
                row=1, col=col_index
            )

        # TRACCIA 2: CAMBIATI (Opacità piena)
        if not df_changed.empty:
            fig.add_trace(
                go.Choropleth(
                    locations=df_changed["country"],
                    locationmode="country names",
                    z=df_changed["cluster"],
                    colorscale=colorscale,
                    zmin=0, zmax=n_clusters - 1,
                    showscale=show_colorbar, # Solo una legenda alla fine
                    marker_opacity=0.9, # <--- ALTA OPACITÀ per evidenziare
                    marker_line_color='black', # Bordo più netto
                    marker_line_width=1.5,
                    # Custom Data per hover
                    customdata=np.stack(
                        [df_changed["capital"], df_changed["country"], df_changed["cluster"]],
                        axis=-1
                    ),
                    hovertemplate=(
                        "<b>%{customdata[0]}</b> (%{customdata[1]})<br>"
                        "Cluster: <b>%{customdata[2]}</b><br>"
                        "<extra></extra>"
                    ),
                    colorbar=dict(
                        title="Cluster",
                        tickvals=list(range(n_clusters)),
                        len=0.8,
                        x=1.02 # Posizionamento a destra
                    ) if show_colorbar else None
                ),
                row=1, col=col_index
            )

    # Aggiungi tracce mappa sinistra
    add_traces(df1_stable, df1_changed, col_index=1, show_colorbar=False)
    
    # Aggiungi tracce mappa destra
    add_traces(df2_stable, df2_changed, col_index=2, show_colorbar=True)

    map_layout_settings = dict(
        showframe=False,
        showcoastlines=True, coastlinecolor="#d1d1d1", # Linea costiera
        showcountries=True, countrycolor="#d1d1d1",   # Confini nazionali (grigio chiaro)
        countrywidth=1.0,
        showland=True, landcolor="#f5f5f5",            # Sfondo terre emerse (grigio chiarissimo)
        showocean=True, oceancolor="white",            # Sfondo oceano (opzionale)
        projection_type="natural earth"
    )

    # 4. Layout Finale
    fig.update_layout(
        geo=map_layout_settings,
        geo2=map_layout_settings,
        width=1400,
        height=600,
        margin={"r":0,"t":50,"l":0,"b":0}
    )

    return fig

def plot_cluster_flow_sankey_first_to_last(df_clusters_dict, first_year, last_year, n_clusters=5):
    """
    Crea un diagramma Sankey che mostra come le capitali si spostano tra cluster tra due anni
    """
    
    my_colors = get_colors()
    
    df1 = df_clusters_dict[first_year][['capital', 'cluster']]
    df2 = df_clusters_dict[last_year][['capital', 'cluster']]
    
    # Merge per trovare i movimenti
    flows = df1.merge(df2, on='capital', suffixes=('_from', '_to'))
    
    # Conta i flussi
    flow_counts = flows.groupby(['cluster_from', 'cluster_to']).size().reset_index(name='count')
    
    # Prepara i dati per Sankey
    sources = []
    targets = []
    values = []
    colors = []
    
    # Colori per i link
    link_colors = {
        'stable': 'rgba(44, 160, 44, 0.6)',      # Verde foresta, più solido
        'change': 'rgba(255, 140, 0, 0.25)'      # Arancione, più trasparente per non coprire il resto
    }
    
    for _, row in flow_counts.iterrows():
        src = int(row['cluster_from'])
        tgt = int(row['cluster_to'])

        sources.append(src)
        targets.append(tgt + n_clusters)
        values.append(row['count'])
        
        # Colora diversamente chi rimane nello stesso cluster
        if src == tgt:
            colors.append(link_colors['stable'])
        else:
            colors.append(link_colors['change'])
    
    # Crea labels
    final_labels = []
    for i in range(n_clusters):
        final_labels.append(f"<b>C{i}</b>")

    # Duplichiamo le label per i nodi di destinazione (stessi nomi)
    labels = final_labels + final_labels
    
    # Colori nodi
    node_colors = [my_colors[i % len(my_colors)] for i in range(n_clusters)] * 2

    # Plot
    fig = go.Figure(data=[go.Sankey(
        arrangement='snap',
        node=dict(
            pad=30,             # Aumentato spazio verticale tra nodi
            thickness=30,       # Nodi leggermente più spessi
            line=dict(color="white", width=0.5), # Bordo bianco per pulizia
            label=labels,
            color=node_colors,
            # ### STYLE 3: Testo Nodi ###
            # Personalizziamo il font delle etichette interne
            hoverinfo='all'
        ),
        link=dict(
            source=sources,
            target=targets,
            value=values,
            color=colors,
            # Aggiungiamo un template hover per i flussi
            hovertemplate='%{value} capitali da %{source.label} a %{target.label}<extra></extra>'
        )
    )])
    
    fig.add_annotation(x=0, y=1.08, xref="paper", yref="paper", text=f"<b>Anno {first_year}</b>", showarrow=False, font=dict(size=16))
    fig.add_annotation(x=1, y=1.08, xref="paper", yref="paper", text=f"<b>Anno {last_year}</b>", showarrow=False, font=dict(size=16))

    fig.update_layout(
        title_text=f"Flussi Climatici",
        font=dict(size=14, family="Arial"), # Font più pulito
        height=700,
        width=1200,
        plot_bgcolor='white',
        margin=dict(t=100, b=50, l=50, r=50) # Margini per far respirare le annotazioni
    )
            
    return fig

# Caricamento
df_clusters_anni, cluster_means_anni = load_data()
aggregation_per_year = load_raw_data()
years = aggregation_per_year['year'].unique()
capitals = aggregation_per_year['capital'].unique()

if df_clusters_anni is not None and cluster_means_anni is not None and aggregation_per_year is not None:
    
    # ==============================================================================
    # SIDEBAR
    # ==============================================================================
    st.sidebar.title("Navigazione")
    page = st.sidebar.radio("Vai a:", ["Mappa Annuale", "Confronto Anni", "Analisi Capitali (Radar)"])
    
    st.sidebar.markdown("---")
    st.sidebar.info(
        "**Nota:** Il clustering riguarda le *capitali*. "
        "La mappa colora direttamente lo stato per una leggibilità piu semplice della mappa."
    )

    # ==============================================================================
    # PAGINA 1: MAPPA ANIMATA
    # ==============================================================================
    if page == "Mappa Annuale":
        st.title("🗺️ Evoluzione Climatica Globale")
        
        st.info("Premi 'Play' per vedere come cambiano i cluster climatici nel tempo.")
        
        # Generazione Mappa
        with st.spinner("Preparazione animazione..."):
            fig_anim = plot_world_slider_streamlit_go(
                df=df_clusters_anni, 
                n_clusters=5
            )
            
            if fig_anim:
                st.plotly_chart(fig_anim, width='stretch')
        
        st.markdown("### 📊 Statistiche Cluster per anno")
        col_stats_1, col_stats_2 = st.columns([1, 1])
        with col_stats_1:
            stats_year = st.slider(
                "Anno statistiche",
                min_value=min(df_clusters_anni.keys()),
                max_value=max(df_clusters_anni.keys()),
                value=max(df_clusters_anni.keys()),
                key="stats_year_slider"
            )
        with col_stats_2:
            stats_view = st.radio(
                "Visualizzazione",
                ["Centroidi Cluster", "Distribuzione Capitali"],
                horizontal=True,
                key="stats_view_radio"
            )

        if stats_view == "Centroidi Cluster":
            st.dataframe(
                cluster_means_anni[stats_year],
                width='stretch'
            )
            st.caption("Valori medi per ciascun cluster nell'anno selezionato.")
        else:
            df_year = df_clusters_anni[stats_year].copy()
            distribuzione = (
                df_year.groupby("cluster")
                .agg(
                    capitali=("capital", "count"),
                )
                .reset_index()
                .rename(columns={"cluster": "Cluster"})
            )
            distribuzione["% Capitali"] = (
                (distribuzione["capitali"] / distribuzione["capitali"].sum()) * 100
            ).round(2)
            st.dataframe(distribuzione, width='stretch')
            st.caption("Conteggio capitali per cluster nell'anno selezionato.")

    # ==============================================================================
    # PAGINA 2: CONFRONTO DUE ANNI
    # ==============================================================================
    elif page == "Confronto Anni":
        st.title("🗺️ Confronto biennale")
        
        col1, col2 = st.columns(2)

        # Selezione 1
        with col1:
            st.subheader("Selezione 1")
            year1 = st.selectbox("Anno 1", sorted(years), key="y1", index=len(years)-1)
            
        # Selezione 2
        with col2:
            st.subheader("Selezione 2")
            year2 = st.selectbox("Anno 2", sorted(years), key="y2", index=len(years)-1)
        
        # Genera il grafico
        fig_compare = plot_comparison_maps(df_clusters_anni, year1, year2, n_clusters=5)
        st.plotly_chart(fig_compare, width='stretch')
        
        st.info("💡 **Nota:** I paesi con colori vivaci hanno cambiato cluster tra i due anni selezionati. I paesi sbiaditi sono rimasti stabili.")
        
        st.markdown("### 📊 Flussi di capitali tra cluster")
        fig_sankey = plot_cluster_flow_sankey_first_to_last(df_clusters_anni, year1, year2)
        st.plotly_chart(fig_sankey, width='stretch')

    # ==============================================================================
    # PAGINA 3: RADAR PLOT E CONFRONTO CAPITALI
    # ==============================================================================
    elif page == "Analisi Capitali (Radar)":
        st.title("🕸️ Analisi Dettagliata (Radar Plot)")
        
        col1, col2 = st.columns(2)
        
        # Selezione 1
        with col1:
            st.subheader("Selezione 1")
            city1 = st.selectbox("Capitale 1", sorted(capitals), key="c1")
            year1 = st.selectbox("Anno 1", sorted(years), key="y1", index=len(years)-1)
            
        # Selezione 2
        with col2:
            st.subheader("Selezione 2")
            city2 = st.selectbox("Capitale 2", sorted(capitals), key="c2", index=1)
            year2 = st.selectbox("Anno 2", sorted(years), key="y2", index=len(years)-1)

        # Recupero dati
        data1 = aggregation_per_year[(aggregation_per_year['capital'] == city1) & (aggregation_per_year['year'] == year1)].iloc[0]
        data2 = aggregation_per_year[(aggregation_per_year['capital'] == city2) & (aggregation_per_year['year'] == year2)].iloc[0]
      
        # Recupero centroidi dei cluster di appartenenza
        cluster_assignment = df_clusters_anni[year1]
        cluster_centroids = cluster_means_anni[year1]
        cluster1_idx = cluster_assignment[cluster_assignment['capital'] == city1]['cluster'].iloc[0]
        centroid1 = cluster_centroids.loc[cluster1_idx]
        
        cluster_assignment = df_clusters_anni[year2]
        cluster_centroids = cluster_means_anni[year2]
        cluster2_idx = cluster_assignment[cluster_assignment['capital'] == city2]['cluster'].iloc[0]
        centroid2 = cluster_centroids.loc[cluster2_idx]

        to_drop=['capital', 'year']

        data1 = data1.drop(columns=to_drop)
        data2 = data2.drop(columns=to_drop)

        comparison_df = pd.DataFrame(
            [data1, data2],
            index=[f"{city1} — {year1} — {cluster1_idx}", f"{city2} — {year2} — {cluster2_idx}"]
        )
        st.dataframe(comparison_df, width='stretch')

        comparison_df_cluster = pd.DataFrame(
            [centroid1, centroid2],
            index=[f"Cluster {cluster1_idx} di {city1} — {year1}", f"Cluster {cluster2_idx} di {city2} — {year2}"]
        )
        st.dataframe(comparison_df_cluster, width='stretch')

        features_to_use = ['mean_temp', 'std_temp', 'rain_total', 'days_rain', 'snow_total', 'mean_sunshine']
        
        # --- PREPARAZIONE RADAR (NORMALIZZAZIONE) ---
        # Per visualizzare temperature (20) e pioggia (1000) insieme, dobbiamo normalizzare.
        # Usiamo MinMaxScaler fittato su TUTTO il dataset per avere coerenza globale.
        scaler_radar = MinMaxScaler()

        global_scaled = aggregation_per_year[features_to_use].copy()
        for col in ['rain_total', 'snow_total']:
            global_scaled[col] = np.log1p(global_scaled[col].clip(lower=0))
        scaler_radar.fit(global_scaled)

        #con clip(lower=0) se un valore è <0 diventa 0

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
            name=f'{city1} ({year1}) - Cluster {cluster1_idx}',
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
            title=f"{city1} ({year1}) vs Media Cluster {cluster1_idx}"
        )

        fig_city2 = go.Figure()
        fig_city2.add_trace(go.Scatterpolar(
            r=r_city2_closed,
            theta=theta_closed,
            fill='toself',
            name=f'{city2} ({year2}) - Cluster {cluster2_idx}',
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
            title=f"{city2} ({year2}) vs Media Cluster {cluster2_idx}"
        )

        col_rad1, col_rad2 = st.columns(2)
        with col_rad1:
            st.plotly_chart(fig_city1, width='stretch')
        with col_rad2:
            st.plotly_chart(fig_city2, width='stretch')
      
        st.info("ℹ️ I valori nel grafico sono normalizzati (0 = min globale, 1 = max globale) per permettere il confronto tra unità di misura diverse (mm, °C, giorni).")
        