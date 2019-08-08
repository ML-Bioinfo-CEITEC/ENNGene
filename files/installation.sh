# Dependency installation for Deepnet framework

# Create separate environment in Conda
conda create --name deepnet python=3.6
conda activate deepnet

# Install packages
conda install -c conda-forge keras
# pip install tensorflow==1.5.0 to get rid of the deprecation warnings until update to TF 2.0
conda install -c anaconda numpy
conda install -c conda-forge matplotlib

pip install hyperas

