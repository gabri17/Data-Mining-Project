import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

cmap = plt.get_cmap("tab10")
hex_codes = [mcolors.to_hex(cmap(i)) for i in range(cmap.N)]

my_colors = {i: hex_codes[i] for i in range(len(hex_codes))}

def plot_world_anno(df_cluster_assignment, year, n_clusters=5, method="K-Means", colors_to_use=my_colors):
    """
    Visualizza la mappa mondiale per un anno specifico
    """
    import geopandas as gpd
    
    # World map data
    world_url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    world = gpd.read_file(world_url)
    
    # Merge data
    world_clusters = world.merge(
        df_cluster_assignment,
        how="left",
        left_on="NAME",
        right_on="country"
    )
    
    fig, ax = plt.subplots(1, 1, figsize=(18, 9))
    
    colors_list = [colors_to_use[i] for i in range(n_clusters)]
    custom_cmap = ListedColormap(colors_list)

    world_clusters.plot(
        column="cluster",
        categorical=True,
        cmap=custom_cmap,
        linewidth=0.5,
        edgecolor="black",
        legend=True,
        ax=ax,
        missing_kwds={
            "color": "whitesmoke",
            "label": "No data"
        }
    )
    
    ax.set_title(
        f"Climate Clusters - {year} ({method})",
        fontsize=16
    )
    
    ax.axis("off")
    plt.show()
