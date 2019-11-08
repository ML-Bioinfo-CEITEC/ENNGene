# Dependency installation for Deepnet framework

# Create separate environment in Conda
conda create --name deepnet python=3.6
conda activate deepnet

# Install Tensorflow 2.0 for deep learning
pip install tensorflow-gpu

# Install other packages
conda install -c conda-forge matplotlib
conda install -c bioconda viennarna

pip install talos
