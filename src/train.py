import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import LinkNeighborLoader
import numpy as np
import pandas as pd
import os
import random
from time import perf_counter
from tqdm import tqdm

# Ensure reproducibility
torch.manual_seed(42)
np.random.seed(42)

def format_time(seconds):
    """Helper to format time nicely."""
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    return f"{int(h):02d}:{int(m):02d}:{int(s):02d}"

def calculate_rmse(pred, target):
    """Root Mean Square Error for Rating Prediction."""
    criterion = nn.MSELoss()
    return torch.sqrt(criterion(pred, target)).item()

def fit(
    model,
    data,
    epochs=50,
    lr=0.001,
    batch_size=1024,
    num_neighbors=[10, 5], # Sample 10 neighbors at hop 1, 5 at hop 2
    optimizer_name="adamW",
    device: torch.device = None,
    checkpoint_dir="checkpoints",
    experiment_name="hetero_gat_v1"
):
    """
    Main training loop for the Heterogeneous GNN.
    
    Args:
        model: The PyG model (MovieHeteroGAT)
        data: The HeteroData object
        epochs: Number of training epochs
        lr: Learning rate
        batch_size: Number of edges (ratings) per batch
        num_neighbors: Sampling strategy for NeighborLoader
        optimizer_name: 'adam', 'adamW', 'sgd'
    """
    
    # 1. Setup Device & Optimizer
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    optimizers = {
        'adam': optim.Adam,
        'adamW': optim.AdamW,
        'sgd': optim.SGD,
    }
    
    # Note: We filter parameters to only optimize those that require gradients
    # (Good practice for GNNs where some embeddings might be frozen)
    optimizer = optimizers.get(optimizer_name, optim.AdamW)(
        filter(lambda p: p.requires_grad, model.parameters()), 
        lr=lr, 
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=epochs, 
        eta_min=lr * 0.01
    )

    # 2. Setup Data Loaders (The "Supervoxel" equivalent for Graphs)
    # We use LinkNeighborLoader because we want to predict RATINGS (edges),
    # not classify users.
    print("Initializing Graph Loaders... (This maps the topology)")
    
    # Edge type to train on: ('user', 'rates', 'movie')
    edge_label_index = data['user', 'rates', 'movie'].edge_index
    edge_label = data['user', 'rates', 'movie'].edge_attr

    # Validate edge attributes
    if edge_label.dim() == 2:
        edge_label = edge_label.squeeze(1)
    assert edge_label.dim() == 1, f"Edge labels should be 1D, got shape {edge_label.shape}"

    # Validate rating range
    min_rating, max_rating = edge_label.min().item(), edge_label.max().item()
    print(f"Rating range in data: [{min_rating:.2f}, {max_rating:.2f}]")
    assert 0.5 <= min_rating <= max_rating <= 5.0, \
        f"Ratings outside expected range [0.5, 5.0]: [{min_rating}, {max_rating}]"

    # Split indices (Reproducible 80/10/10 split)
    num_edges = edge_label_index.size(1)
    split_generator = torch.Generator().manual_seed(42)
    perm = torch.randperm(num_edges, generator=split_generator)

    train_idx = perm[:int(0.8 * num_edges)]
    val_idx = perm[int(0.8 * num_edges):int(0.9 * num_edges)]
    test_idx = perm[int(0.9 * num_edges):]

    print(f"Split sizes - Train: {len(train_idx):,}, Val: {len(val_idx):,}, Test: {len(test_idx):,}")

    train_loader = LinkNeighborLoader(
        data,
        num_neighbors=num_neighbors, 
        edge_label_index=(('user', 'rates', 'movie'), edge_label_index[:, train_idx]),
        edge_label=edge_label[train_idx],
        batch_size=batch_size,
        shuffle=True,
        neg_sampling_ratio=0.0, # We are doing regression on existing edges, not link prediction
        num_workers=4
    )

    val_loader = LinkNeighborLoader(
        data,
        num_neighbors=num_neighbors,
        edge_label_index=(('user', 'rates', 'movie'), edge_label_index[:, val_idx]),
        edge_label=edge_label[val_idx],
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    test_loader = LinkNeighborLoader(
        data,
        num_neighbors=num_neighbors,
        edge_label_index=(('user', 'rates', 'movie'), edge_label_index[:, test_idx]),
        edge_label=edge_label[test_idx],
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )

    print(f"Training on {len(train_loader)} batches. Validation on {len(val_loader)} batches. Test on {len(test_loader)} batches.")

    # 3. Stats Tracking
    stats = {
        'epoch': [],
        'train_loss': [],
        'train_rmse': [],
        'val_loss': [],
        'val_rmse': []
    }
    
    best_val_rmse = float('inf')
    start_time = perf_counter()
    loss_fn = nn.MSELoss() # Regression Loss

    # ==========================
    # TRAINING LOOP
    # ==========================
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}\n{'='*60}")
        epoch_start = perf_counter()
        
        # --- TRAIN STEP ---
        model.train()
        train_loss_accum = 0
        train_rmse_accum = 0
        
        # tqdm for progress bar
        pbar = tqdm(train_loader, desc=f"Train Epoch {epoch+1}")
        
        for batch in pbar:
            batch = batch.to(device)
            optimizer.zero_grad()

            # A. Forward Pass (Get Node Embeddings)
            # Create x_dict with global node IDs from the batch
            x_dict = {
                'user': batch['user'].n_id,
                'movie': batch['movie'].n_id
            }

            x_dict_out = model(x_dict, batch.edge_index_dict, batch.edge_attr_dict)

            # B. Decode (Predict Ratings for the specific edges in this batch)
            # LinkNeighborLoader puts the target edges under the edge type key
            edge_label_index = batch['user', 'rates', 'movie'].edge_label_index
            pred_ratings = model.decode(x_dict_out, edge_label_index)

            # C. Loss Calculation
            # batch.edge_label contains the actual ratings
            target = batch['user', 'rates', 'movie'].edge_label.squeeze() 
            
            loss = loss_fn(pred_ratings, target)
            loss.backward()
            
            # Gradient Clipping (Critical for GATs to prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            
            optimizer.step()

            # Logging
            current_loss = loss.item()
            current_rmse = torch.sqrt(loss).item()
            train_loss_accum += current_loss
            train_rmse_accum += current_rmse
            
            pbar.set_postfix({'loss': f"{current_loss:.4f}", 'rmse': f"{current_rmse:.4f}"})

        scheduler.step()
        
        avg_train_loss = train_loss_accum / len(train_loader)
        avg_train_rmse = train_rmse_accum / len(train_loader)

        # --- VALIDATION STEP ---
        avg_val_loss, avg_val_rmse = validate(model, val_loader, device, loss_fn)

        # --- EPOCH SUMMARY ---
        epoch_time = perf_counter() - epoch_start
        print(f"\nEnd of Epoch {epoch+1}")
        print(f"Time: {format_time(epoch_time)}")
        print(f"Train RMSE: {avg_train_rmse:.4f} | Val RMSE: {avg_val_rmse:.4f}")

        # Update Stats
        stats['epoch'].append(epoch + 1)
        stats['train_loss'].append(avg_train_loss)
        stats['train_rmse'].append(avg_train_rmse)
        stats['val_loss'].append(avg_val_loss)
        stats['val_rmse'].append(avg_val_rmse)

        # Checkpointing
        if avg_val_rmse < best_val_rmse:
            best_val_rmse = avg_val_rmse
            save_model(model, optimizer, epoch, experiment_name, checkpoint_dir, best=True)
            print(f"⭐ New Best Model Saved! (RMSE: {best_val_rmse:.4f})")
        
        # Regular save
        if (epoch + 1) % 5 == 0:
            save_model(model, optimizer, epoch, experiment_name, checkpoint_dir, best=False)

    # Final Save & Stats
    save_stats_to_csv(stats, experiment_name, checkpoint_dir)
    print(f"\nTraining Complete. Total Time: {format_time(perf_counter() - start_time)}")


def validate(model, loader, device, loss_fn):
    """
    Validation loop.
    """
    model.eval()
    total_loss = 0
    total_rmse = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating", leave=False):
            batch = batch.to(device)

            # Create x_dict with global node IDs from the batch
            x_dict = {
                'user': batch['user'].n_id,
                'movie': batch['movie'].n_id
            }

            # Forward
            x_dict_out = model(x_dict, batch.edge_index_dict, batch.edge_attr_dict)

            # Decode
            edge_label_index = batch['user', 'rates', 'movie'].edge_label_index
            pred_ratings = model.decode(x_dict_out, edge_label_index)

            # Loss
            target = batch['user', 'rates', 'movie'].edge_label.squeeze()
            loss = loss_fn(pred_ratings, target)
            
            total_loss += loss.item()
            total_rmse += torch.sqrt(loss).item()
            
    return total_loss / len(loader), total_rmse / len(loader)


def save_model(model, optimizer, epoch, name, checkpoint_dir, best=False):
    """Saves model state."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    filename = f"{name}_best.pth" if best else f"{name}_epoch_{epoch+1}.pth"
    path = os.path.join(checkpoint_dir, filename)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def save_stats_to_csv(stats, name, checkpoint_dir):
    """Saves training history to CSV."""
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"{name}_stats.csv")
    
    df = pd.DataFrame(stats)
    df.to_csv(path, index=False)
    print(f"Stats saved to {path}")