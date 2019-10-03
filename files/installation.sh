# Dependency installation for Deepnet framework

# Create separate environment in Conda
conda create --name deepnet python=3.6
conda activate deepnet

# Install packages
conda install -c conda-forge keras
conda install -c anaconda numpy
conda install -c conda-forge matplotlib
conda install -c bioconda viennarna

pip install talos
