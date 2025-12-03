"""
Configuration dataclasses for the movie recommendation GNN.
"""
from dataclasses import dataclass, field
from typing import List, Tuple
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ModelConfig:
    """Configuration for the MovieHeteroGAT model architecture."""

    embedding_dim: int = 64
    hidden_channels: int = 64
    out_channels: int = 64
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.5
    edge_dim: int = 1  # Number of edge features (rating is 1 feature)
    activation: str = 'leaky_relu'  # 'leaky_relu', 'relu', or 'elu'
    weight_init: str = 'kaiming'  # 'kaiming' or 'xavier'
    rating_decoder: str = 'mlp'  # 'mlp' or 'dot'
    rating_range: Tuple[float, float] = (0.5, 5.0)  # Min and max rating values


@dataclass
class TrainingConfig:
    """Configuration for training hyperparameters."""

    epochs: int = 50
    lr: float = 0.001  # Learning rate
    batch_size: int = 1024  # Number of edges per batch
    num_neighbors: List[int] = field(default_factory=lambda: [10, 5])  # Neighbor sampling per hop
    optimizer: str = "adamW"  # 'adam', 'adamW', or 'sgd'
    weight_decay: float = 1e-4
    grad_clip_max_norm: float = 2.0  # Maximum gradient norm for clipping
    scheduler: str = "cosine"  # 'cosine', 'step', or 'none'
    scheduler_eta_min: float = 1e-5  # Minimum learning rate for cosine annealing


@dataclass
class DataConfig:
    """Configuration for data loading and preprocessing."""

    dataset_path: str = field(default_factory=lambda: os.getenv('DATASET_PATH', ''))
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42
    num_workers: int = 4  # Number of workers for data loaders

    def __post_init__(self):
        """Validate configuration after initialization."""
        if not self.dataset_path:
            raise ValueError(
                "DATASET_PATH not set. Please set it in your .env file or "
                "provide it directly in the config."
            )

        ratio_sum = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(ratio_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Train/val/test ratios must sum to 1.0, got {ratio_sum}"
            )


@dataclass
class ExperimentConfig:
    """Configuration for experiment tracking and checkpointing."""

    checkpoint_dir: str = "checkpoints"
    experiment_name: str = "hetero_gat_v1"
    save_every_n_epochs: int = 5  # Save checkpoint every N epochs
    save_best_only: bool = True  # Only save when validation improves
    metric_to_track: str = "val_rmse"  # Metric for determining "best" model
    log_dir: str = "logs"  # Directory for training logs
    use_wandb: bool = False  # Whether to use Weights & Biases logging
    wandb_project: str = "movie-recommendation-gnn"
    wandb_entity: str = ""  # Leave empty to use default

    def __post_init__(self):
        """Create directories if they don't exist."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)


@dataclass
class Config:
    """Complete configuration combining all sub-configs."""

    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    def summary(self) -> str:
        """Return a formatted summary of the configuration."""
        lines = [
            "="*70,
            "CONFIGURATION SUMMARY",
            "="*70,
            "",
            "Model Configuration:",
            f"  Architecture: HeteroGATv2",
            f"  Layers: {self.model.num_layers}",
            f"  Attention Heads: {self.model.num_heads}",
            f"  Embedding Dim: {self.model.embedding_dim}",
            f"  Hidden Channels: {self.model.hidden_channels}",
            f"  Output Channels: {self.model.out_channels}",
            f"  Dropout: {self.model.dropout}",
            f"  Rating Decoder: {self.model.rating_decoder.upper()}",
            f"  Rating Range: [{self.model.rating_range[0]}, {self.model.rating_range[1]}]",
            "",
            "Training Configuration:",
            f"  Epochs: {self.training.epochs}",
            f"  Learning Rate: {self.training.lr}",
            f"  Batch Size: {self.training.batch_size}",
            f"  Optimizer: {self.training.optimizer}",
            f"  Weight Decay: {self.training.weight_decay}",
            f"  Gradient Clipping: {self.training.grad_clip_max_norm}",
            f"  Neighbor Sampling: {self.training.num_neighbors}",
            "",
            "Data Configuration:",
            f"  Dataset: {self.data.dataset_path}",
            f"  Train/Val/Test Split: {self.data.train_ratio}/{self.data.val_ratio}/{self.data.test_ratio}",
            f"  Random Seed: {self.data.random_seed}",
            f"  Num Workers: {self.data.num_workers}",
            "",
            "Experiment Configuration:",
            f"  Name: {self.experiment.experiment_name}",
            f"  Checkpoint Dir: {self.experiment.checkpoint_dir}",
            f"  Log Dir: {self.experiment.log_dir}",
            f"  Save Every N Epochs: {self.experiment.save_every_n_epochs}",
            f"  Track Metric: {self.experiment.metric_to_track}",
            "="*70
        ]
        return "\n".join(lines)


def load_config_from_dict(config_dict: dict) -> Config:
    """
    Load configuration from a dictionary (e.g., parsed from YAML/JSON).

    Args:
        config_dict: Dictionary containing configuration values

    Returns:
        Config object

    Example:
        >>> config_dict = {
        ...     'model': {'num_layers': 3, 'num_heads': 8},
        ...     'training': {'epochs': 100, 'lr': 0.0005}
        ... }
        >>> config = load_config_from_dict(config_dict)
    """
    model_config = ModelConfig(**config_dict.get('model', {}))
    training_config = TrainingConfig(**config_dict.get('training', {}))
    data_config = DataConfig(**config_dict.get('data', {}))
    experiment_config = ExperimentConfig(**config_dict.get('experiment', {}))

    return Config(
        model=model_config,
        training=training_config,
        data=data_config,
        experiment=experiment_config
    )


if __name__ == "__main__":
    """Test configuration loading and display."""
    config = Config()
    print(config.summary())
