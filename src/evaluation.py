"""
Evaluation metrics for the movie recommendation system.
"""
import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import numpy as np
from tqdm import tqdm


def calculate_rmse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate Root Mean Squared Error.

    Args:
        predictions: Predicted ratings
        targets: Actual ratings

    Returns:
        RMSE value
    """
    mse = torch.mean((predictions - targets) ** 2)
    return torch.sqrt(mse).item()


def calculate_mae(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate Mean Absolute Error.

    Args:
        predictions: Predicted ratings
        targets: Actual ratings

    Returns:
        MAE value
    """
    mae = torch.mean(torch.abs(predictions - targets))
    return mae.item()


def calculate_mse(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """
    Calculate Mean Squared Error.

    Args:
        predictions: Predicted ratings
        targets: Actual ratings

    Returns:
        MSE value
    """
    mse = torch.mean((predictions - targets) ** 2)
    return mse.item()


def calculate_precision_at_k(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int = 10,
    threshold: float = 4.0
) -> float:
    """
    Calculate Precision@K for recommendation ranking.

    Measures what fraction of top-K recommendations are actually relevant.

    Args:
        predictions: Predicted ratings for all items [N]
        targets: Actual ratings for all items [N]
        k: Number of top recommendations to consider
        threshold: Minimum rating to be considered "relevant"

    Returns:
        Precision@K value
    """
    # Get top-K predictions
    _, top_k_indices = torch.topk(predictions, k)

    # Check which of these are actually relevant (rating >= threshold)
    relevant_in_top_k = (targets[top_k_indices] >= threshold).float().sum()

    return (relevant_in_top_k / k).item()


def calculate_recall_at_k(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int = 10,
    threshold: float = 4.0
) -> float:
    """
    Calculate Recall@K for recommendation ranking.

    Measures what fraction of all relevant items are in the top-K.

    Args:
        predictions: Predicted ratings for all items [N]
        targets: Actual ratings for all items [N]
        k: Number of top recommendations to consider
        threshold: Minimum rating to be considered "relevant"

    Returns:
        Recall@K value
    """
    # Get top-K predictions
    _, top_k_indices = torch.topk(predictions, k)

    # Total number of relevant items
    total_relevant = (targets >= threshold).float().sum()

    if total_relevant == 0:
        return 0.0

    # How many relevant items are in top-K
    relevant_in_top_k = (targets[top_k_indices] >= threshold).float().sum()

    return (relevant_in_top_k / total_relevant).item()


def calculate_ndcg_at_k(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int = 10
) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain at K.

    NDCG measures the quality of a ranking by comparing it to the ideal ranking,
    with higher-ranked relevant items contributing more to the score.

    Args:
        predictions: Predicted ratings for all items [N]
        targets: Actual ratings for all items [N]
        k: Number of top recommendations to consider

    Returns:
        NDCG@K value between 0 and 1
    """
    # Get top-K predictions
    _, top_k_indices = torch.topk(predictions, k)

    # Actual ratings for predicted top-K items
    actual_ratings = targets[top_k_indices]

    # Calculate DCG (Discounted Cumulative Gain)
    # DCG = sum(rel_i / log2(i + 1)) where i is the position (1-indexed)
    positions = torch.arange(1, k + 1, dtype=torch.float32, device=predictions.device)
    dcg = torch.sum(actual_ratings / torch.log2(positions + 1))

    # Calculate IDCG (Ideal DCG) - what we'd get with perfect ranking
    ideal_ratings, _ = torch.topk(targets, min(k, len(targets)))
    ideal_positions = torch.arange(1, len(ideal_ratings) + 1, dtype=torch.float32, device=predictions.device)
    idcg = torch.sum(ideal_ratings / torch.log2(ideal_positions + 1))

    # Normalize
    if idcg == 0:
        return 0.0

    return (dcg / idcg).item()


def calculate_hit_rate_at_k(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k: int = 10,
    threshold: float = 4.0
) -> float:
    """
    Calculate Hit Rate@K.

    Measures whether at least one relevant item appears in the top-K.
    Returns 1 if there's a hit, 0 otherwise.

    Args:
        predictions: Predicted ratings for all items [N]
        targets: Actual ratings for all items [N]
        k: Number of top recommendations to consider
        threshold: Minimum rating to be considered "relevant"

    Returns:
        Hit rate (0 or 1)
    """
    # Get top-K predictions
    _, top_k_indices = torch.topk(predictions, k)

    # Check if any of these are relevant
    relevant_in_top_k = (targets[top_k_indices] >= threshold).any()

    return float(relevant_in_top_k.item())


@torch.no_grad()
def evaluate_model(
    model,
    data_loader,
    device: torch.device,
    k_values: List[int] = [5, 10, 20],
    rating_threshold: float = 4.0
) -> Dict[str, float]:
    """
    Comprehensive evaluation of the model on a data loader.

    Args:
        model: The trained MovieHeteroGAT model
        data_loader: DataLoader with test/validation data
        device: Device to run evaluation on
        k_values: List of K values for ranking metrics
        rating_threshold: Threshold for relevant items in ranking metrics

    Returns:
        Dictionary of metric names and values
    """
    model.eval()

    all_predictions = []
    all_targets = []

    # Collect all predictions
    for batch in tqdm(data_loader, desc="Evaluating", leave=False):
        batch = batch.to(device)

        # Create x_dict with global node IDs from the batch
        x_dict = {
            'user': batch['user'].n_id,
            'movie': batch['movie'].n_id
        }

        # Forward pass
        x_dict_out = model(x_dict, batch.edge_index_dict, batch.edge_attr_dict)

        # Decode predictions
        edge_label_index = batch['user', 'rates', 'movie'].edge_label_index
        pred_ratings = model.decode(x_dict_out, edge_label_index)

        # Collect
        all_predictions.append(pred_ratings.cpu())
        all_targets.append(batch['user', 'rates', 'movie'].edge_label.cpu())

    # Concatenate all batches
    predictions = torch.cat(all_predictions)
    targets = torch.cat(all_targets)

    # Calculate metrics
    metrics = {}

    # Regression metrics
    metrics['rmse'] = calculate_rmse(predictions, targets)
    metrics['mae'] = calculate_mae(predictions, targets)
    metrics['mse'] = calculate_mse(predictions, targets)

    # Ranking metrics (calculated per-user if we had user info, here globally)
    for k in k_values:
        metrics[f'precision@{k}'] = calculate_precision_at_k(
            predictions, targets, k, rating_threshold
        )
        metrics[f'recall@{k}'] = calculate_recall_at_k(
            predictions, targets, k, rating_threshold
        )
        metrics[f'ndcg@{k}'] = calculate_ndcg_at_k(predictions, targets, k)
        metrics[f'hit_rate@{k}'] = calculate_hit_rate_at_k(
            predictions, targets, k, rating_threshold
        )

    return metrics


def print_evaluation_results(metrics: Dict[str, float], title: str = "Evaluation Results"):
    """
    Pretty-print evaluation metrics.

    Args:
        metrics: Dictionary of metric names and values
        title: Title for the results section
    """
    print("\n" + "="*60)
    print(f"{title:^60}")
    print("="*60)

    # Regression metrics
    print("\nRegression Metrics:")
    if 'rmse' in metrics:
        print(f"  RMSE: {metrics['rmse']:.4f}")
    if 'mae' in metrics:
        print(f"  MAE:  {metrics['mae']:.4f}")
    if 'mse' in metrics:
        print(f"  MSE:  {metrics['mse']:.4f}")

    # Ranking metrics
    k_values = sorted(set(
        int(key.split('@')[1])
        for key in metrics.keys()
        if '@' in key
    ))

    if k_values:
        print("\nRanking Metrics:")
        for k in k_values:
            print(f"\n  K={k}:")
            if f'precision@{k}' in metrics:
                print(f"    Precision@{k}: {metrics[f'precision@{k}']:.4f}")
            if f'recall@{k}' in metrics:
                print(f"    Recall@{k}:    {metrics[f'recall@{k}']:.4f}")
            if f'ndcg@{k}' in metrics:
                print(f"    NDCG@{k}:      {metrics[f'ndcg@{k}']:.4f}")
            if f'hit_rate@{k}' in metrics:
                print(f"    Hit Rate@{k}:  {metrics[f'hit_rate@{k}']:.4f}")

    print("="*60 + "\n")


if __name__ == "__main__":
    """Test evaluation metrics."""

    # Create dummy predictions and targets
    torch.manual_seed(42)
    predictions = torch.randn(100) * 2 + 3  # Mean ~3, range ~[1, 5]
    targets = torch.randint(1, 6, (100,)).float()

    print("Testing Regression Metrics:")
    print(f"RMSE: {calculate_rmse(predictions, targets):.4f}")
    print(f"MAE: {calculate_mae(predictions, targets):.4f}")
    print(f"MSE: {calculate_mse(predictions, targets):.4f}")

    print("\nTesting Ranking Metrics:")
    for k in [5, 10, 20]:
        print(f"\nK={k}:")
        print(f"  Precision@{k}: {calculate_precision_at_k(predictions, targets, k):.4f}")
        print(f"  Recall@{k}: {calculate_recall_at_k(predictions, targets, k):.4f}")
        print(f"  NDCG@{k}: {calculate_ndcg_at_k(predictions, targets, k):.4f}")
        print(f"  Hit Rate@{k}: {calculate_hit_rate_at_k(predictions, targets, k):.4f}")
