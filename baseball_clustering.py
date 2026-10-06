# This file was adapted from a ipynb that I wrote on Google Colab
# After adapting it from ipynb I wrote some more functionality in VSCode before uploading this product to a GitHub repo
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('baseball_data.csv')

# Base offensive stats to cluster on
stat_cols = ['K_plus', 'BB_plus', 'OBP_plus', 'SLG_plus', 'wRC_plus']
matching_features = ['player_name', 'age', 'year_ID', 'team_ID'] + stat_cols

# Filter for 100 pa to remove small sample size noise
df2 = df[df['pa'] >= 100][matching_features].copy()
df2.dropna(inplace=True)

# Sort sequentially by player and year so diff() calculates correctly
df2 = df2.sort_values(by=['player_name', 'year_ID']).reset_index(drop=True)

# Calculate the year-over-year gap to check for missed seasons
# This will be 1 if they played consecutive seasons, or >1 if they missed a year
df2['year_diff'] = df2.groupby('player_name')['year_ID'].diff()

# Calculate 1-year deltas for all offensive stats
delta_cols = []
for col in stat_cols:
    delta_col = f'{col}_delta_1yr'
    delta_cols.append(delta_col)
    
    # Calculate the raw difference from the previous row
    raw_diff = df2.groupby('player_name')[col].diff()
    
    # If year_diff is not exactly 1 (they missed a season or it's their rookie year),
    # set the delta to NaN so we don't calculate a false multi-year jump.
    df2[delta_col] = raw_diff.where(df2['year_diff'] == 1.0)

# Drop rows with NaN deltas (rookies and first year back from a missed season)
df2.dropna(subset=delta_cols, inplace=True)
df2.drop(columns=['year_diff'], inplace=True)
df2.reset_index(drop=True, inplace=True)

# Define final features: Base Stats (how good they are) + Deltas (which way they are trending)
all_features = stat_cols + delta_cols
X = df2[all_features].copy()

# Scale the feature set
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit K-Means
kmeans = KMeans(n_clusters=8, max_iter=5000, random_state=42).fit(X_scaled)
df2['Cluster'] = kmeans.labels_

# Commenting all this out because this was the graphing I did on the python notebook
'''
# Compress 7 features down to 2 summary dimensions
pca = PCA(n_components=2)
pca_components = pca.fit_transform(X)

# Add coordinates to dataframe
df2['PCA_X'] = pca_components[:, 0]
df2['PCA_Y'] = pca_components[:, 1]

# Plot the clusters
plt.figure(figsize=(10, 8))
sns.scatterplot(
    data=df2,
    x='PCA_X',
    y='PCA_Y',
    hue='Cluster',
    palette='tab20',
    s=100,
    alpha=0.7
)

plt.title("MLB Hitter Player Archetypes (PCA)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title='Cluster')
plt.grid(True, linestyle='--', alpha=0.5)
plt.show()
'''

def find_similar_players(player_name, year, df, scaled_stats, top_n=30):
    # Filter for the specific player-season
    target_match = df[(df['player_name'] == player_name) & (df['year_ID'] == year)]
    
    if target_match.empty:
        raise ValueError(f"Player '{player_name}' for season {year} not found in the dataset.")

    target_cluster = target_match['Cluster'].values[0]

    # Filter down to players in the target cluster
    cluster_mask = (df['Cluster'] == target_cluster).values
    cluster_df = df[cluster_mask].reset_index(drop=True)
    cluster_scaled_data = scaled_stats[cluster_mask]

    # Get n+1 nearest neighbors (this naturally includes the player matching themselves)
    nn = NearestNeighbors(n_neighbors=top_n + 1, metric='cosine')
    nn.fit(cluster_scaled_data)

    # Locate target season within the cluster slice
    target_index = cluster_df[(cluster_df['player_name'] == player_name) & (cluster_df['year_ID'] == year)].index[0]
    target_stats = cluster_scaled_data[target_index].reshape(1, -1)

    distances, indices = nn.kneighbors(target_stats)

    # KEEP all indices, including index 0 (the target player)
    all_indices = indices[0]
    all_distances = distances[0]

    desired_columns = ['player_name', 'year_ID', 'OBP_plus', 'SLG_plus', 'wRC_plus']
    
    # Slice the dataframe to grab the target player and all neighbors
    results_df = cluster_df.iloc[all_indices][desired_columns].copy()
    results_df['distance'] = all_distances
    
    # Reset index to guarantee the target player is strictly at index 0
    results_df.reset_index(drop=True, inplace=True)
    
    return results_df


# Old prediction function, predicts exact values instead of probabilities
def calculate_projections(similar_players_df, full_df, stat_cols=['OBP_plus', 'SLG_plus', 'wRC_plus']):
    # Extract the target player from the top row
    target_player = similar_players_df.iloc[0]
    
    # Extract the remaining rows as the actual doppelgangers
    neighbors_df = similar_players_df.iloc[1:].copy()
    
    # Create a lookup key for the subsequent season (Year N + 1)
    neighbors_df['next_year_ID'] = neighbors_df['year_ID'] + 1

    # Join against full_df to grab their actual Year N + 1 stats
    lookup_cols = ['player_name', 'year_ID'] + stat_cols
    next_season_stats = full_df[lookup_cols].copy()

    merged = neighbors_df.merge(
        next_season_stats,
        left_on=['player_name', 'next_year_ID'],
        right_on=['player_name', 'year_ID'],
        suffixes=('', '_next')
    )

    # Filter out dead ends (missing stats or no Year N + 1 record)
    next_stat_cols = [f'{col}_next' for col in stat_cols]
    valid_neighbors = merged.dropna(subset=next_stat_cols).copy()

    if valid_neighbors.empty:
        raise ValueError(f"None of the similar players for {target_player['player_name']} have recorded stats in Year N+1.")
    
    # Calculate weights of valid neighbors
    valid_neighbors['weight'] = 1 / (valid_neighbors['distance'] + 0.001)
    weight_sum = valid_neighbors['weight'].sum()

    # Compute the weighted average for each projected stat
    projections = {
        'target_player': target_player['player_name'],
        'base_year': target_player['year_ID'],
        'projected_year': target_player['year_ID'] + 1
    }
    
    for col in stat_cols:
        weighted_stat = (valid_neighbors[f'{col}_next'] * valid_neighbors['weight']).sum()
        projections[f'proj_{col}'] = round(weighted_stat / weight_sum, 1)

    projections['valid_neighbors_count'] = len(valid_neighbors)
    
    return projections



def calculate_full_projections(similar_players_df, full_df, stat_cols=['OBP_plus', 'SLG_plus', 'wRC_plus']):
    target_player = similar_players_df.iloc[0]
    neighbors_df = similar_players_df.iloc[1:].copy()
    neighbors_df['next_year_ID'] = neighbors_df['year_ID'] + 1

    # Join against full_df to grab Year N + 1 stats
    lookup_cols = ['player_name', 'year_ID'] + stat_cols
    next_season_stats = full_df[lookup_cols].copy()

    merged = neighbors_df.merge(
        next_season_stats,
        left_on=['player_name', 'next_year_ID'],
        right_on=['player_name', 'year_ID'],
        suffixes=('', '_next')
    )

    next_stat_cols = [f'{col}_next' for col in stat_cols]
    valid_neighbors = merged.dropna(subset=next_stat_cols).copy()

    if valid_neighbors.empty:
        raise ValueError(f"No valid neighbors with Year N+1 stats found.")

    # Calculate similarity weights
    valid_neighbors['weight'] = 1 / (valid_neighbors['distance'] + 0.001)
    total_weight = valid_neighbors['weight'].sum()

    # Initialize the output dictionary
    projections = {
        'target_player': target_player['player_name'],
        'base_year': target_player['year_ID'],
        'projected_year': target_player['year_ID'] + 1,
        'valid_neighbors': len(valid_neighbors)
    }
    
    for col in stat_cols:
        target_col = f'{col}_next'
        base_value = target_player[col]
        
        # Percentile Projections (Floor, Median, Ceiling)
        sorted_neighbors = valid_neighbors.sort_values(by=target_col)
        cumulative_weight_pct = sorted_neighbors['weight'].cumsum() / total_weight
        
        projections[f'{col}_Floor_10th'] = round(sorted_neighbors.loc[cumulative_weight_pct >= 0.10, target_col].iloc[0], 1)
        projections[f'{col}_Median_50th'] = round(sorted_neighbors.loc[cumulative_weight_pct >= 0.50, target_col].iloc[0], 1)
        projections[f'{col}_Ceiling_90th'] = round(sorted_neighbors.loc[cumulative_weight_pct >= 0.90, target_col].iloc[0], 1)

        # Dynamic Probabilities
        # Did the neighbors improve at all?
        improve_weight = valid_neighbors.loc[valid_neighbors[target_col] > base_value, 'weight'].sum()
        
        # Did the neighbors improve by at least 20%?
        breakout_weight = valid_neighbors.loc[valid_neighbors[target_col] >= (base_value * 1.20), 'weight'].sum()
        
        # Did the neighbors decline by at least 20%?
        collapse_weight = valid_neighbors.loc[valid_neighbors[target_col] <= (base_value * 0.80), 'weight'].sum()
        
        projections[f'{col}_Prob_Improve'] = f"{round((improve_weight / total_weight) * 100, 1)}%"
        projections[f'{col}_Prob_Breakout'] = f"{round((breakout_weight / total_weight) * 100, 1)}%"
        projections[f'{col}_Prob_Collapse'] = f"{round((collapse_weight / total_weight) * 100, 1)}%"

    return projections


dg_df = find_similar_players('Francisco Lindor', 2024, df2, X_scaled, top_n=30)
print(dg_df)
p_proj = calculate_full_projections(dg_df, df2)
for key, value in p_proj.items():
    print(f"{key}: {value}")

