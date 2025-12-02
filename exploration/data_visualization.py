from matplotlib.ticker import ScalarFormatter
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_graph_stats(df_ratings, df_films, title_prefix="", min_ratings_per_movie=0):
    """
    Draws the 4 standard plots for the Movie Graph Analysis.
    """
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle(f"{title_prefix} Dataset Overview", fontsize=16)

    # --- Plot A: Rating Distribution ---
    sns.histplot(data=df_ratings, x='rating', bins=10, kde=False, ax=axes[0, 0], color='teal')
    axes[0, 0].set_title('Global Rating Distribution')
    axes[0, 0].set_xlabel('Rating Score')
    axes[0, 0].set_ylabel('Count')

    # --- Plot B: Ratings per User (Log Axis) ---
    user_counts = df_ratings['user_name'].value_counts()
    sns.histplot(user_counts, log_scale=True, ax=axes[0, 1], color='orange')
    axes[0, 1].set_title('Ratings per User (Log Axis)')
    axes[0, 1].set_xlabel('Number of Ratings')
    axes[0, 1].set_ylabel('Number of Users')
    axes[0, 1].axvline(20, color='red', linestyle='--', label='Min 20 Ratings')
    axes[0, 1].xaxis.set_major_formatter(ScalarFormatter())
    axes[0, 1].set_xticks([1, 10, 100, 1000, 10000])
    axes[0, 1].legend()

    # --- Plot C: Ratings per Movie (Log Axis) ---
    movie_counts = df_ratings['film_id'].value_counts()
    sns.histplot(movie_counts, log_scale=True, ax=axes[1, 0], color='purple')
    axes[1, 0].set_title('Ratings per Movie (Log Axis)')
    axes[1, 0].set_xlabel('Number of Ratings')
    axes[1, 0].set_ylabel('Number of Movies')
    axes[1, 0].xaxis.set_major_formatter(ScalarFormatter())
    # Adjust ticks dynamically based on max count
    ticks = [1, 5, 10, 100, 1000, 10000, 100000]
    axes[1, 0].set_xticks([t for t in ticks if t < movie_counts.max()])
    
    # Add text annotation about the filter
    axes[1, 0].text(0.05, 0.9, f"Min Ratings Filter: {min_ratings_per_movie}", 
                    transform=axes[1, 0].transAxes, 
                    bbox=dict(facecolor='white', alpha=0.8))

    # --- Plot D: Release Year ---
    if 'year' in df_films.columns:
        # We only care about films that actually exist in our filtered ratings
        relevant_films = df_films[df_films['film_id'].isin(df_ratings['film_id'].unique())]
        years = relevant_films['year'].dropna()
        years = years[(years > 1900) & (years < 2026)] 
        sns.histplot(years, bins=50, ax=axes[1, 1], color='steelblue')
        axes[1, 1].set_title('Movies by Release Year (Filtered)')
    else:
        axes[1, 1].text(0.5, 0.5, 'Year column not found', ha='center')

    plt.tight_layout()
    plt.show()
    
    # --- Print Stats ---
    n_users = df_ratings['user_name'].nunique()
    n_items = df_ratings['film_id'].nunique()
    n_ratings = len(df_ratings)
    if n_users * n_items > 0:
        sparsity = 1 - (n_ratings / (n_users * n_items))
    else:
        sparsity = 0

    print(f"\n{'='*20} {title_prefix.upper()} STATS {'='*20}")
    print(f"Unique Users: {n_users:,}")
    print(f"Unique Movies: {n_items:,}")
    print(f"Total Interactions: {n_ratings:,}")
    print(f"Sparsity: {sparsity:.6%}")