#!/bin/bash

ENV_NAME="movie-recommendation"

echo "=========================================="
echo "   RE-ATTEMPTING: The Pip-Only Strategy   "
echo "=========================================="

# 1. Clean Clean Clean
# We remove the environment to ensure no "cpu_openblas" ghosts remain.
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "🗑️  Removing broken environment..."
    conda env remove -n "$ENV_NAME" -y
fi

# 2. Create Minimal Conda Base
# We ONLY ask Conda for Python. We do NOT ask it for PyTorch.
echo "🌱 Creating minimal Python 3.11 environment..."
conda create -n "$ENV_NAME" python=3.11 -y

# 3. Activate
eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

# 4. Install PyTorch (The GPU Version) via PIP
# We point explicitly to the CUDA 12.1 wheel index. 
# Pip will download the massive binary that includes EVERYTHING it needs.
echo "🚀 Force-installing PyTorch 2.5.1 (CUDA 12.1) via Pip..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Install PyG Extensions
# Now that we have a guaranteed CUDA PyTorch, we install the matching extensions.
echo "🔗 Installing PyG Extensions..."
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.5.1+cu121.html

# 6. Install Main Libraries
echo "📦 Installing PyG and Data Science Stack..."
pip install torch_geometric
pip install numpy pandas matplotlib seaborn tqdm python-dotenv scikit-learn

# 7. Verification
echo "=========================================="
echo "🔍 VERIFYING INSTALLATION..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
python -c "import torch_geometric; print(f'PyG Version: {torch_geometric.__version__}')"
echo "=========================================="