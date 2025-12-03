"""
Inference utilities for the movie recommendation system.
"""
import torch
from torch_geometric.data import HeteroData
from typing import List, Tuple, Optional, Dict
import os
from src.model import MovieHeteroGAT


def load_trained_model(
    checkpoint_path: str,
    data: HeteroData,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[MovieHeteroGAT, dict]:
    """
    Load a trained model from a checkpoint.

    Args:
        checkpoint_path: Path to the .pth checkpoint file
        data: HeteroData object (needed for metadata and node counts)
        device: Device to load model onto

    Returns:
        Tuple of (model, checkpoint_info)
            - model: Loaded MovieHeteroGAT model
            - checkpoint_info: Dictionary with epoch, optimizer state, etc.

    Raises:
        FileNotFoundError: If checkpoint doesn't exist
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")

    print(f"Loading model from {checkpoint_path}...")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract model configuration from checkpoint or use defaults
    # Note: In production, you'd want to save the config with the model
    model_config = checkpoint.get('model_config', {})

    # Create model
    model = MovieHeteroGAT(
        metadata=data.metadata(),
        num_users=data['user'].num_nodes,
        num_movies=data['movie'].num_nodes,
        **model_config
    )

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    checkpoint_info = {
        'epoch': checkpoint.get('epoch', -1),
        'optimizer_state_dict': checkpoint.get('optimizer_state_dict'),
    }

    print(f"✓ Model loaded successfully (epoch {checkpoint_info['epoch']})")

    return model, checkpoint_info


@torch.no_grad()
def predict_rating(
    model: MovieHeteroGAT,
    data: HeteroData,
    user_id: int,
    movie_id: int,
    device: torch.device
) -> float:
    """
    Predict rating for a single user-movie pair.

    Args:
        model: Trained MovieHeteroGAT model
        data: Full HeteroData object
        user_id: User node ID (0-indexed)
        movie_id: Movie node ID (0-indexed)
        device: Device to run inference on

    Returns:
        Predicted rating value
    """
    model.eval()

    # Create a simple batch with just these two nodes
    # For full message passing, we'd need to sample neighbors
    # Here we do a simplified version

    # Get all node IDs (for full graph)
    user_ids = torch.arange(data['user'].num_nodes, device=device)
    movie_ids = torch.arange(data['movie'].num_nodes, device=device)

    # Prepare input
    x_dict = {
        'user': user_ids,
        'movie': movie_ids
    }

    # Move data to device
    edge_index_dict = {
        key: data[key].edge_index.to(device)
        for key in data.edge_types
    }

    edge_attr_dict = None
    if 'edge_attr' in data['user', 'rates', 'movie']:
        edge_attr = data['user', 'rates', 'movie'].edge_attr
        if edge_attr.dim() == 2:
            edge_attr = edge_attr.squeeze(1)
        edge_attr_dict = {
            ('user', 'rates', 'movie'): edge_attr.to(device)
        }

    # Forward pass
    x_dict_out = model(x_dict, edge_index_dict, edge_attr_dict)

    # Decode for this specific pair
    edge_label_index = torch.tensor([[user_id], [movie_id]], device=device)
    prediction = model.decode(x_dict_out, edge_label_index)

    return prediction.item()


@torch.no_grad()
def batch_predict(
    model: MovieHeteroGAT,
    data: HeteroData,
    user_movie_pairs: List[Tuple[int, int]],
    device: torch.device,
    batch_size: int = 10000
) -> List[float]:
    """
    Predict ratings for multiple user-movie pairs efficiently.

    Args:
        model: Trained MovieHeteroGAT model
        data: Full HeteroData object
        user_movie_pairs: List of (user_id, movie_id) tuples
        device: Device to run inference on
        batch_size: Maximum number of pairs to process at once

    Returns:
        List of predicted ratings
    """
    model.eval()

    all_predictions = []

    # Get all node embeddings once
    user_ids = torch.arange(data['user'].num_nodes, device=device)
    movie_ids = torch.arange(data['movie'].num_nodes, device=device)

    x_dict = {
        'user': user_ids,
        'movie': movie_ids
    }

    edge_index_dict = {
        key: data[key].edge_index.to(device)
        for key in data.edge_types
    }

    edge_attr_dict = None
    if 'edge_attr' in data['user', 'rates', 'movie']:
        edge_attr = data['user', 'rates', 'movie'].edge_attr
        if edge_attr.dim() == 2:
            edge_attr = edge_attr.squeeze(1)
        edge_attr_dict = {
            ('user', 'rates', 'movie'): edge_attr.to(device)
        }

    # Get embeddings
    x_dict_out = model(x_dict, edge_index_dict, edge_attr_dict)

    # Batch decode
    for i in range(0, len(user_movie_pairs), batch_size):
        batch_pairs = user_movie_pairs[i:i + batch_size]

        # Create edge_label_index for this batch
        users = torch.tensor([pair[0] for pair in batch_pairs], device=device)
        movies = torch.tensor([pair[1] for pair in batch_pairs], device=device)
        edge_label_index = torch.stack([users, movies])

        # Decode
        predictions = model.decode(x_dict_out, edge_label_index)
        all_predictions.extend(predictions.cpu().tolist())

    return all_predictions


@torch.no_grad()
def get_top_k_recommendations(
    model: MovieHeteroGAT,
    data: HeteroData,
    user_id: int,
    k: int = 10,
    exclude_rated: bool = True,
    device: torch.device = None
) -> Tuple[List[int], List[float]]:
    """
    Get top-K movie recommendations for a user.

    Args:
        model: Trained MovieHeteroGAT model
        data: Full HeteroData object
        user_id: User node ID (0-indexed)
        k: Number of recommendations to return
        exclude_rated: Whether to exclude movies the user has already rated
        device: Device to run inference on

    Returns:
        Tuple of (movie_ids, predicted_ratings)
            - movie_ids: List of recommended movie IDs
            - predicted_ratings: List of predicted ratings for those movies
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()

    # Get all node embeddings
    user_ids = torch.arange(data['user'].num_nodes, device=device)
    movie_ids = torch.arange(data['movie'].num_nodes, device=device)

    x_dict = {
        'user': user_ids,
        'movie': movie_ids
    }

    edge_index_dict = {
        key: data[key].edge_index.to(device)
        for key in data.edge_types
    }

    edge_attr_dict = None
    if 'edge_attr' in data['user', 'rates', 'movie']:
        edge_attr = data['user', 'rates', 'movie'].edge_attr
        if edge_attr.dim() == 2:
            edge_attr = edge_attr.squeeze(1)
        edge_attr_dict = {
            ('user', 'rates', 'movie'): edge_attr.to(device)
        }

    # Forward pass to get embeddings
    x_dict_out = model(x_dict, edge_index_dict, edge_attr_dict)

    # Predict ratings for this user with ALL movies
    num_movies = data['movie'].num_nodes
    users_repeated = torch.full((num_movies,), user_id, device=device)
    all_movies = torch.arange(num_movies, device=device)
    edge_label_index = torch.stack([users_repeated, all_movies])

    predictions = model.decode(x_dict_out, edge_label_index)

    # Exclude already rated movies if requested
    if exclude_rated:
        # Find movies this user has rated
        edge_index = data['user', 'rates', 'movie'].edge_index
        user_mask = edge_index[0] == user_id
        rated_movies = edge_index[1][user_mask]

        # Set their predictions to -inf so they won't be selected
        predictions[rated_movies] = float('-inf')

    # Get top-K
    top_k_scores, top_k_indices = torch.topk(predictions, k)

    return top_k_indices.cpu().tolist(), top_k_scores.cpu().tolist()


def save_model_with_config(
    model: MovieHeteroGAT,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    save_path: str,
    model_config: Optional[Dict] = None,
    metrics: Optional[Dict] = None
):
    """
    Save model checkpoint with configuration and metrics.

    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch number
        save_path: Path to save checkpoint
        model_config: Dictionary of model configuration
        metrics: Dictionary of evaluation metrics
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }

    if model_config is not None:
        checkpoint['model_config'] = model_config

    if metrics is not None:
        checkpoint['metrics'] = metrics

    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")


if __name__ == "__main__":
    """Test inference functions."""
    from src.data_loader import load_hetero_data
    from dotenv import load_dotenv
    import os

    load_dotenv()

    # Load data
    dataset_path = os.getenv('DATASET_PATH', '')
    if not dataset_path:
        print("Error: DATASET_PATH not set")
        exit(1)

    print("Loading data...")
    data = load_hetero_data(dataset_path)

    print("\nInference utilities loaded successfully!")
    print("To test, train a model first and provide a checkpoint path.")
