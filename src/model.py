import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, HeteroConv, Linear

class MovieHeteroGAT(nn.Module):
    def __init__(
        self,
        metadata,
        num_users: int,
        num_movies: int,
        embedding_dim: int = 64,
        hidden_channels: int = 64,
        out_channels: int = 64,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.5,
        edge_dim: int = 1,  # We have 1 edge feature: The rating
        activation: str = 'leaky_relu',
        weight_init: str = 'kaiming',
        rating_decoder: str = 'mlp',  # 'mlp' or 'dot'
        rating_range: tuple = (0.5, 5.0)  # min and max rating
    ):
        """
        A flexible Heterogeneous GAT model.
        
        Args:
            metadata: Tuple (node_types, edge_types) from hetero_data.metadata()
            num_users: Count of unique users for embedding table
            num_movies: Count of unique movies for embedding table
            embedding_dim: Size of initial learnable embeddings
            hidden_channels: Size of GNN hidden layers
            out_channels: Size of final output embedding
            num_layers: Number of message passing layers
            num_heads: Number of attention heads for GAT
            edge_dim: Dimensionality of edge attributes (e.g. rating)
        """
        super().__init__()
        
        # 1. Configuration & Logging
        self.metadata = metadata
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation_name = activation
        
        # Store training history (matching your style)
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        
        # 2. Initial Embeddings (The "Features")
        # Since we mapped IDs to integers, we need to learn a vector for each ID.
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.movie_emb = nn.Embedding(num_movies, embedding_dim)
        
        # Project embeddings to hidden_channels if they differ
        self.lin_user = Linear(embedding_dim, hidden_channels)
        self.lin_movie = Linear(embedding_dim, hidden_channels)

        # 3. Message Passing Layers (The "Convs")
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            # We determine input/output dims for this specific layer
            in_dim = hidden_channels * num_heads if i > 0 else hidden_channels
            out_dim = hidden_channels if i < num_layers - 1 else out_channels
            
            # Create a HeteroConv. This wraps multiple GAT layers (one per edge type)
            # into a single "layer" that handles the whole bipartite graph.
            conv_dict = {}
            for edge_type in metadata[1]:
                # edge_type example: ('user', 'rates', 'movie')
                conv_dict[edge_type] = GATv2Conv(
                    in_channels=(-1, -1), # Auto-infer source/target dims
                    out_channels=out_dim,
                    heads=num_heads,
                    concat=True if i < num_layers - 1 else False, # Concat in hidden, Avg in final
                    edge_dim=edge_dim, # Inject edge features (ratings) into attention
                    add_self_loops=False, # Critical for Bipartite graphs
                    dropout=dropout
                )
            
            self.convs.append(HeteroConv(conv_dict, aggr='sum'))
            
            # Layer Norm for stability (Optional but recommended for Deep GNNs)
            # Note: We need a dictionary of norms, one per node type
            norm_dict = nn.ModuleDict()
            for node_type in metadata[0]:
                norm_size = out_dim * num_heads if i < num_layers - 1 else out_dim
                norm_dict[node_type] = nn.LayerNorm(norm_size)
            self.norms.append(norm_dict)

        # 4. Rating Decoder Configuration
        self.rating_decoder_type = rating_decoder
        self.rating_min, self.rating_max = rating_range
        self.rating_scale = self.rating_max - self.rating_min

        # Create decoder based on type
        if rating_decoder == 'mlp':
            self.rating_decoder = nn.Sequential(
                nn.Linear(out_channels * 2, 64),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
                nn.Sigmoid()  # Output in [0, 1]
            )
        else:
            # Use dot product + sigmoid (no extra parameters)
            self.rating_decoder = None

        # 5. Initialize Weights
        self.apply_weight_init(weight_init)

        self.model_details = (
            f"HeteroGATv2_L{num_layers}_H{num_heads}_Emb{embedding_dim}_"
            f"Hidden{hidden_channels}_Out{out_channels}_{weight_init}_"
            f"Decoder{rating_decoder.upper()}"
        )

    def apply_weight_init(self, init_type):
        """Applies custom weight initialization to Linear and Embedding layers."""
        def init_func(m):
            if isinstance(m, (nn.Linear, nn.Embedding)):
                if init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                elif init_type == 'xavier':
                    nn.init.xavier_uniform_(m.weight)
                # Note: GATv2Conv has its own internal init, usually Glorot
        
        self.apply(init_func)

    def get_activation(self, x):
        if self.activation_name == 'leaky_relu':
            return F.leaky_relu(x, negative_slope=0.2)
        elif self.activation_name == 'relu':
            return F.relu(x)
        elif self.activation_name == 'elu':
            return F.elu(x)
        return x

    def forward(self, x_dict, edge_index_dict, edge_attr_dict=None):
        """
        Forward pass through the heterogeneous GAT layers.

        Args:
            x_dict: Dictionary containing node ID tensors for embedding lookup
                    e.g., {'user': tensor([0, 5, 12, ...]), 'movie': tensor([1, 8, 23, ...])}
                    These should be GLOBAL node IDs that will be used to index embeddings.
            edge_index_dict: The connectivity (local batch node IDs after sampling)
            edge_attr_dict: Edge features (ratings) to use in attention computation

        Returns:
            Dictionary of node embeddings after message passing
        """

        # 1. Prepare Initial Node Features
        # x_dict contains the GLOBAL node IDs for nodes present in this batch
        # We use these to look up embeddings, then project to hidden space
        user_ids = x_dict['user']  # Global IDs
        movie_ids = x_dict['movie']  # Global IDs

        # Lookup embeddings and project to hidden space
        x_dict_out = {}
        x_dict_out['user'] = self.lin_user(self.user_emb(user_ids))
        x_dict_out['movie'] = self.lin_movie(self.movie_emb(movie_ids))
        
        # 2. Message Passing Loop
        for i, conv in enumerate(self.convs):
            # A. Convolution (Message Passing)
            x_dict_out = conv(x_dict_out, edge_index_dict, edge_attr_dict)
            
            # B. Apply Norm & Activation & Dropout (per node type)
            for node_type, x in x_dict_out.items():
                x = self.norms[i][node_type](x)
                x = self.get_activation(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                x_dict_out[node_type] = x
        
        return x_dict_out

    def decode(self, x_dict, edge_label_index):
        """
        Predicts ratings for specific user-movie pairs.

        Args:
            x_dict: Dictionary of node embeddings (output from forward pass)
            edge_label_index: [2, N] tensor of (user, movie) pairs to predict ratings for
                             Uses LOCAL batch node IDs (0 to num_nodes_in_batch - 1)

        Returns:
            Predicted ratings in range [rating_min, rating_max] (default [0.5, 5.0])
        """
        # Get embeddings for the source (users) and destination (movies) pairs
        # edge_label_index uses local batch IDs (after message passing)
        users = x_dict['user'][edge_label_index[0]]
        movies = x_dict['movie'][edge_label_index[1]]

        if self.rating_decoder_type == 'mlp':
            # Concatenate user and movie embeddings and pass through MLP
            combined = torch.cat([users, movies], dim=-1)
            normalized_rating = self.rating_decoder(combined).squeeze(-1)
        else:
            # Dot product + sigmoid
            dot_product = (users * movies).sum(dim=-1)
            normalized_rating = torch.sigmoid(dot_product)

        # Scale from [0, 1] to [rating_min, rating_max]
        return normalized_rating * self.rating_scale + self.rating_min
    
    @torch.no_grad()
    def recommend_for_new_user(self, liked_movie_indices, all_movie_indices, k=10):
        """
        BASELINE cold-start recommendation using simple embedding averaging.

        ⚠ IMPORTANT: This method does NOT use message passing. It simply averages
        movie embeddings to create a user vector, bypassing the GNN's learned
        graph structure. This is a simple baseline approach.

        For production cold-start scenarios, consider:
        1. Using movie content features (synopsis, director, genre, cast)
        2. Popularity-based fallback recommendations
        3. Running message passing with temporary edges (computationally expensive)
        4. Hybrid approach: content-based filtering + this method

        Args:
            liked_movie_indices (Tensor): IDs of movies the user likes (e.g., [50, 102, 999])
            all_movie_indices (Tensor): IDs of all candidate movies to rank
            k (int): Number of recommendations to return

        Returns:
            top_indices: Indices of top-k recommended movies
            top_scores: Prediction scores for those movies
        """
        # 1. Get the embeddings of the movies the user LIKES
        # Shape: [num_liked, embedding_dim]
        # We use the projection layer to ensure they match the hidden space
        liked_movie_embs = self.lin_movie(self.movie_emb(liked_movie_indices))
        
        # 2. Create the "User Vector" by averaging the movies they like
        # This approximates what the GNN would have learned if this user existed
        # Shape: [1, embedding_dim]
        user_vector = liked_movie_embs.mean(dim=0, keepdim=True)
        
        # 3. Get all candidate movie embeddings
        # Shape: [num_candidates, embedding_dim]
        candidate_embs = self.lin_movie(self.movie_emb(all_movie_indices))
        
        # 4. Calculate Scores (Dot Product)
        # Shape: [1, num_candidates]
        scores = (user_vector * candidate_embs).sum(dim=-1)
        
        # 5. Rank
        top_scores, top_indices = torch.topk(scores, k)
        
        return top_indices, top_scores