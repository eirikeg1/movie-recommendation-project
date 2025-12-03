"""
Data loading and preprocessing utilities for the movie recommendation GNN.
"""
import torch
from torch_geometric.data import HeteroData
import os
from typing import Tuple, Dict


def load_hetero_data(path: str, device: str = 'cpu') -> HeteroData:
    """
    Load preprocessed HeteroData object from disk.

    Args:
        path: Path to the .pt file containing the HeteroData object
        device: Device to load data onto ('cpu' or 'cuda')

    Returns:
        HeteroData object with user-movie bipartite graph

    Raises:
        FileNotFoundError: If the data file doesn't exist
        RuntimeError: If the loaded object is not a valid HeteroData
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Data file not found at {path}. "
            f"Please ensure the dataset has been preprocessed and saved."
        )

    print(f"Loading HeteroData from {path}...")

    # Load with appropriate device mapping
    if device == 'cuda' and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available. Loading to CPU.")
        device = 'cpu'

    data = torch.load(path, map_location=device)

    # Validate it's a HeteroData object
    if not isinstance(data, HeteroData):
        raise RuntimeError(
            f"Expected HeteroData object, got {type(data)}. "
            f"Please check the data file."
        )

    print(f"✓ Loaded HeteroData successfully")

    # Run validation
    validate_data_structure(data)

    return data


def validate_data_structure(data: HeteroData) -> None:
    """
    Validate that the HeteroData object has the expected structure.

    Args:
        data: HeteroData object to validate

    Raises:
        AssertionError: If validation fails
    """
    print("\nValidating data structure...")

    # Check node types
    assert 'user' in data.node_types, "Missing 'user' node type"
    assert 'movie' in data.node_types, "Missing 'movie' node type"
    print(f"✓ Node types: {data.node_types}")

    # Check edge types
    expected_edge = ('user', 'rates', 'movie')
    assert expected_edge in data.edge_types, f"Missing edge type {expected_edge}"
    print(f"✓ Edge types: {data.edge_types}")

    # Validate node counts
    num_users = data['user'].num_nodes
    num_movies = data['movie'].num_nodes
    assert num_users > 0, "No user nodes found"
    assert num_movies > 0, "No movie nodes found"
    print(f"✓ Users: {num_users:,} | Movies: {num_movies:,}")

    # Validate edge structure
    edge_index = data['user', 'rates', 'movie'].edge_index
    assert edge_index.shape[0] == 2, f"Edge index should be [2, N], got {edge_index.shape}"
    num_edges = edge_index.shape[1]
    assert num_edges > 0, "No edges found"
    print(f"✓ Edges (ratings): {num_edges:,}")

    # Check edge index is within bounds
    max_user_id = edge_index[0].max().item()
    max_movie_id = edge_index[1].max().item()
    assert max_user_id < num_users, \
        f"Edge references user ID {max_user_id} but only {num_users} users exist"
    assert max_movie_id < num_movies, \
        f"Edge references movie ID {max_movie_id} but only {num_movies} movies exist"
    print(f"✓ Edge indices within bounds")

    # Validate edge attributes (ratings)
    if 'edge_attr' in data['user', 'rates', 'movie']:
        edge_attr = data['user', 'rates', 'movie'].edge_attr

        # Ensure proper shape
        if edge_attr.dim() == 2:
            assert edge_attr.shape[1] == 1, \
                f"Edge attr should be [N, 1] for ratings, got {edge_attr.shape}"
        elif edge_attr.dim() == 1:
            pass  # [N] is also acceptable
        else:
            raise ValueError(f"Edge attr has unexpected shape: {edge_attr.shape}")

        # Validate rating range
        min_rating = edge_attr.min().item()
        max_rating = edge_attr.max().item()
        print(f"✓ Rating range: [{min_rating:.2f}, {max_rating:.2f}]")

        # Check if ratings are in expected range (Letterboxd: 0.5-5.0)
        if not (0.5 <= min_rating <= max_rating <= 5.0):
            print(f"⚠ Warning: Ratings outside expected range [0.5, 5.0]")
    else:
        print("⚠ Warning: No edge attributes found (expected ratings)")

    # Calculate and display sparsity
    sparsity = 1 - (num_edges / (num_users * num_movies))
    print(f"✓ Graph sparsity: {sparsity:.4%}")

    print("✓ Data validation passed!\n")


def get_data_statistics(data: HeteroData) -> Dict[str, any]:
    """
    Calculate and return detailed statistics about the dataset.

    Args:
        data: HeteroData object

    Returns:
        Dictionary containing various statistics
    """
    edge_index = data['user', 'rates', 'movie'].edge_index
    num_users = data['user'].num_nodes
    num_movies = data['movie'].num_nodes
    num_edges = edge_index.shape[1]

    # Calculate per-user and per-movie rating counts
    user_counts = torch.bincount(edge_index[0], minlength=num_users)
    movie_counts = torch.bincount(edge_index[1], minlength=num_movies)

    stats = {
        'num_users': num_users,
        'num_movies': num_movies,
        'num_interactions': num_edges,
        'sparsity': 1 - (num_edges / (num_users * num_movies)),
        'avg_ratings_per_user': user_counts.float().mean().item(),
        'median_ratings_per_user': user_counts.float().median().item(),
        'avg_ratings_per_movie': movie_counts.float().mean().item(),
        'median_ratings_per_movie': movie_counts.float().median().item(),
        'min_ratings_per_user': user_counts.min().item(),
        'max_ratings_per_user': user_counts.max().item(),
        'min_ratings_per_movie': movie_counts.min().item(),
        'max_ratings_per_movie': movie_counts.max().item(),
    }

    # Add rating statistics if available
    if 'edge_attr' in data['user', 'rates', 'movie']:
        edge_attr = data['user', 'rates', 'movie'].edge_attr
        if edge_attr.dim() == 2:
            edge_attr = edge_attr.squeeze(1)

        stats['rating_min'] = edge_attr.min().item()
        stats['rating_max'] = edge_attr.max().item()
        stats['rating_mean'] = edge_attr.mean().item()
        stats['rating_std'] = edge_attr.std().item()

    return stats


def print_data_statistics(data: HeteroData) -> None:
    """
    Print formatted statistics about the dataset.

    Args:
        data: HeteroData object
    """
    stats = get_data_statistics(data)

    print("\n" + "="*60)
    print("DATASET STATISTICS")
    print("="*60)
    print(f"Users:              {stats['num_users']:>12,}")
    print(f"Movies:             {stats['num_movies']:>12,}")
    print(f"Interactions:       {stats['num_interactions']:>12,}")
    print(f"Sparsity:           {stats['sparsity']:>12.4%}")
    print()
    print(f"Avg ratings/user:   {stats['avg_ratings_per_user']:>12.1f}")
    print(f"Median ratings/user:{stats['median_ratings_per_user']:>12.0f}")
    print(f"Min ratings/user:   {stats['min_ratings_per_user']:>12,}")
    print(f"Max ratings/user:   {stats['max_ratings_per_user']:>12,}")
    print()
    print(f"Avg ratings/movie:  {stats['avg_ratings_per_movie']:>12.1f}")
    print(f"Median ratings/movie:{stats['median_ratings_per_movie']:>12.0f}")
    print(f"Min ratings/movie:  {stats['min_ratings_per_movie']:>12,}")
    print(f"Max ratings/movie:  {stats['max_ratings_per_movie']:>12,}")

    if 'rating_mean' in stats:
        print()
        print(f"Rating range:       [{stats['rating_min']:.1f}, {stats['rating_max']:.1f}]")
        print(f"Rating mean:        {stats['rating_mean']:>12.3f}")
        print(f"Rating std:         {stats['rating_std']:>12.3f}")

    print("="*60 + "\n")


def create_train_val_test_split(
    num_edges: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create reproducible train/val/test splits for edges.

    Args:
        num_edges: Total number of edges to split
        train_ratio: Fraction for training (default: 0.8)
        val_ratio: Fraction for validation (default: 0.1)
        test_ratio: Fraction for testing (default: 0.1)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_idx, val_idx, test_idx) tensors

    Raises:
        AssertionError: If ratios don't sum to 1.0
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    # Create reproducible generator
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_edges, generator=generator)

    # Calculate split indices
    train_end = int(train_ratio * num_edges)
    val_end = int((train_ratio + val_ratio) * num_edges)

    train_idx = perm[:train_end]
    val_idx = perm[train_end:val_end]
    test_idx = perm[val_end:]

    print(f"\nCreated splits:")
    print(f"  Train: {len(train_idx):,} ({len(train_idx)/num_edges:.1%})")
    print(f"  Val:   {len(val_idx):,} ({len(val_idx)/num_edges:.1%})")
    print(f"  Test:  {len(test_idx):,} ({len(test_idx)/num_edges:.1%})")

    return train_idx, val_idx, test_idx


if __name__ == "__main__":
    """Test data loading with the dataset from .env"""
    from dotenv import load_dotenv
    load_dotenv()

    dataset_path = os.getenv('DATASET_PATH', '')
    if not dataset_path:
        print("Error: DATASET_PATH not set in .env file")
        exit(1)

    # Load and validate data
    data = load_hetero_data(dataset_path)

    # Print statistics
    print_data_statistics(data)

    # Test split creation
    num_edges = data['user', 'rates', 'movie'].edge_index.shape[1]
    train_idx, val_idx, test_idx = create_train_val_test_split(num_edges)
