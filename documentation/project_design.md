# Graph-Based Neural Movie Recommendation System

### Technical Design & Data Summary

## 1. Executive Summary

This project implements a **Heterogeneous Graph Neural Network (GNN)** to build a "User-as-Critic" recommendation engine. Unlike traditional matrix factorization approaches that treat user interactions as static entries, this system models the dataset as a bipartite graph of Users and Movies.

The core hypothesis is that high-volume users ("Super Users") act as latent critics. By using **Graph Attention Networks (GATv2)**, the model learns dynamic attention weights between users, effectively finding "critics" whose specific tastes align with the target user for specific genres or contexts, rather than relying on global similarity.

## 2. Dataset Architecture (Letterboxd)

The dataset represents a highly dense subset of the Letterboxd platform, specifically focusing on the most active users. This removes the "Cold Start" problem for users and provides a rich signal for message passing.

### 2.1 Graph Statistics (Post-Filtering)

| Metric | Value | Description | 
 | ----- | ----- | ----- | 
| **Unique Users** | **11,061** | Highly active "Super Users" (Avg \~1,600 ratings/user). | 
| **Unique Movies** | **142,374** | Filtered to exclude noise (min 5 ratings). | 
| **Total Edges** | **17,793,097** | User-Movie interactions. | 
| **Sparsity** | **98.87%** | Extremely dense for a recommendation dataset. | 
| **Edge Attributes** | Rating (0.5 - 5.0) | Explicit feedback signals. | 

### 2.2 Data Topology

The data is modeled as a directed, heterogeneous graph in **PyTorch Geometric**:

* **Node Types:**

  * `User`: Initialized with learnable embeddings.

  * `Movie`: Initialized with learnable embeddings (future: synopsis BERT embeddings).

* **Edge Types:**

  * `(user, rates, movie)`: Forward pass (Taste expression).

  * `(movie, rated_by, user)`: Backward pass (Message aggregation).

## 3. Algorithmic Approach

### 3.1 Model Architecture: HeteroGATv2

The system utilizes **Graph Attention Networks v2 (GATv2)** adapted for heterogeneous graphs.

* **Mechanism:** `GATv2Conv` computes a dynamic attention coefficient $\alpha_{ij}$ for every edge.

  * *Intuition:* If User A and User B both rated *Pulp Fiction*, the GNN computes how relevant User B's opinion is to User A. This allows the model to "listen" to different neighbors depending on the specific movie context.

* **Layer Structure:**

  * **Input Projection:** Linear layers project User and Movie features to a common hidden dimension ($d=64$).

  * **Message Passing (L1 & L2):** Two hops of `HeteroConv` wrappers containing `GATv2Conv`. This allows a user to receive signals from:

    * 1-Hop: Movies they watched.

    * 2-Hop: Other users who watched those movies (Collaborative Filtering).

* **Objective Function:**

  * **Regression:** Minimizing MSE between predicted attention scores and actual user ratings.

  * **Ranking (Optional):** Bayesian Personalized Ranking (BPR) to optimize the relative order of items.

### 3.2 Training Strategy (RTX 3090 Optimized)

Given the graph scale (17M edges), full-batch training is impossible on 24GB VRAM.

* **Sampling:** PyTorch Geometric `NeighborLoader`.

  * Samples small subgraphs (e.g., 1024 users + 10 neighbors per hop) for mini-batch training.

* **Storage:**

  * **System RAM (64GB):** Holds the full graph topology (Adjacency Matrix).

  * **VRAM (24GB):** Holds the active mini-batch and gradients.

* **Optimization:**

  * **Gradient Checkpointing:** Used to trade compute for memory, allowing for larger batch sizes or attention heads.

## 4. Why This Approach?

1. **Solves the "Average Taste" Problem:** Traditional algorithms often regress to the mean. GAT allows the model to focus on specific, highly-correlated neighbors ("Critics") for specific recommendations.

2. **Exploits Density:** The dataset's unique high density (1,600 ratings/user) makes it an ideal candidate for Deep Graph Learning, which typically struggles with sparse data.

3. **Inductive Capability:** The GNN architecture can naturally incorporate side information (Movie Posters, Synopsis, Director) in the future without changing the core architecture.