#!/usr/bin/env python3
"""
Main training script for the movie recommendation GNN.

Usage:
    python train_model.py
    python train_model.py --experiment-name my_experiment
    python train_model.py --epochs 100 --lr 0.0005
"""
import argparse
import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_hetero_data, print_data_statistics
from src.model import MovieHeteroGAT
from src.train import fit
from src.config import Config, ModelConfig, TrainingConfig, DataConfig, ExperimentConfig
from src.evaluation import evaluate_model, print_evaluation_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train a Heterogeneous GAT for movie recommendations'
    )

    # Experiment settings
    parser.add_argument(
        '--experiment-name',
        type=str,
        default='hetero_gat_v1',
        help='Name for this experiment (default: hetero_gat_v1)'
    )

    # Model architecture
    parser.add_argument('--embedding-dim', type=int, default=64)
    parser.add_argument('--hidden-channels', type=int, default=64)
    parser.add_argument('--out-channels', type=int, default=64)
    parser.add_argument('--num-layers', type=int, default=2)
    parser.add_argument('--num-heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument(
        '--rating-decoder',
        type=str,
        default='mlp',
        choices=['mlp', 'dot'],
        help='Type of rating decoder (default: mlp)'
    )

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adamW',
        choices=['adam', 'adamW', 'sgd']
    )

    # Data settings
    parser.add_argument('--dataset-path', type=str, default=None)

    # Device
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu']
    )

    # Evaluation
    parser.add_argument(
        '--eval-test-set',
        action='store_true',
        help='Evaluate on test set after training'
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Create configuration
    config = Config()

    # Override config with command line arguments
    if args.dataset_path:
        config.data.dataset_path = args.dataset_path

    config.experiment.experiment_name = args.experiment_name

    # Model config overrides
    config.model.embedding_dim = args.embedding_dim
    config.model.hidden_channels = args.hidden_channels
    config.model.out_channels = args.out_channels
    config.model.num_layers = args.num_layers
    config.model.num_heads = args.num_heads
    config.model.dropout = args.dropout
    config.model.rating_decoder = args.rating_decoder

    # Training config overrides
    config.training.epochs = args.epochs
    config.training.lr = args.lr
    config.training.batch_size = args.batch_size
    config.training.optimizer = args.optimizer

    # Print configuration
    print(config.summary())

    # Set device
    device = torch.device(args.device)
    print(f"\nUsing device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Load data
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)

    try:
        data = load_hetero_data(config.data.dataset_path, device='cpu')
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease ensure DATASET_PATH is set correctly in your .env file")
        print("or provide --dataset-path argument.")
        sys.exit(1)

    # Print data statistics
    print_data_statistics(data)

    # Initialize model
    print("="*70)
    print("INITIALIZING MODEL")
    print("="*70)

    model = MovieHeteroGAT(
        metadata=data.metadata(),
        num_users=data['user'].num_nodes,
        num_movies=data['movie'].num_nodes,
        embedding_dim=config.model.embedding_dim,
        hidden_channels=config.model.hidden_channels,
        out_channels=config.model.out_channels,
        num_layers=config.model.num_layers,
        num_heads=config.model.num_heads,
        dropout=config.model.dropout,
        edge_dim=config.model.edge_dim,
        activation=config.model.activation,
        weight_init=config.model.weight_init,
        rating_decoder=config.model.rating_decoder,
        rating_range=config.model.rating_range
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel: {model.model_details}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Train
    print("\n" + "="*70)
    print("TRAINING")
    print("="*70)

    try:
        fit(
            model=model,
            data=data,
            epochs=config.training.epochs,
            lr=config.training.lr,
            batch_size=config.training.batch_size,
            num_neighbors=config.training.num_neighbors,
            optimizer_name=config.training.optimizer,
            device=device,
            checkpoint_dir=config.experiment.checkpoint_dir,
            experiment_name=config.experiment.experiment_name
        )
    except KeyboardInterrupt:
        print("\n\n⚠ Training interrupted by user")
        return
    except Exception as e:
        print(f"\n\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n✅ Training completed successfully!")

    # Optional: Evaluate on test set
    if args.eval_test_set:
        print("\n" + "="*70)
        print("EVALUATING ON TEST SET")
        print("="*70)

        from torch_geometric.loader import LinkNeighborLoader

        # Load best model
        best_model_path = Path(config.experiment.checkpoint_dir) / f"{config.experiment.experiment_name}_best.pth"

        if best_model_path.exists():
            print(f"Loading best model from {best_model_path}...")
            checkpoint = torch.load(best_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            print("✓ Model loaded")

            # Create test loader
            edge_label_index = data['user', 'rates', 'movie'].edge_index
            edge_label = data['user', 'rates', 'movie'].edge_attr

            if edge_label.dim() == 2:
                edge_label = edge_label.squeeze(1)

            num_edges = edge_label_index.size(1)
            split_generator = torch.Generator().manual_seed(42)
            perm = torch.randperm(num_edges, generator=split_generator)
            test_idx = perm[int(0.9 * num_edges):]

            test_loader = LinkNeighborLoader(
                data,
                num_neighbors=config.training.num_neighbors,
                edge_label_index=(('user', 'rates', 'movie'), edge_label_index[:, test_idx]),
                edge_label=edge_label[test_idx],
                batch_size=config.training.batch_size,
                shuffle=False,
                num_workers=4
            )

            # Evaluate
            test_metrics = evaluate_model(
                model=model,
                data_loader=test_loader,
                device=device,
                k_values=[5, 10, 20],
                rating_threshold=4.0
            )

            # Print results
            print_evaluation_results(test_metrics, title="Test Set Results")
        else:
            print(f"⚠ Best model not found at {best_model_path}")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)
    print(f"\nCheckpoints saved in: {config.experiment.checkpoint_dir}")
    print(f"Experiment name: {config.experiment.experiment_name}")
    print("\nTo use the trained model:")
    print("  from src.inference import load_trained_model, get_top_k_recommendations")
    print(f"  model, _ = load_trained_model('{best_model_path}', data)")
    print("  movie_ids, ratings = get_top_k_recommendations(model, data, user_id=0, k=10)")


if __name__ == '__main__':
    main()
