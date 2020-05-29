## deepnet App 
Deepnet App (*name to be defined*) aims to facilitate application of deep neural networks on genomics data without the need of programming.

### Installation
#### PyPI
(*Installation from PyPI currently not available due to missing dependencies.*)

#### Conda environment
If you do not have [Anaconda](https://www.anaconda.com/distribution/) installed on your computer, please do so first. 
- Download the latest version of the app from [the repository](https://gitlab.com/RBP_Bioinformatics/deepnet/-/tags)
- Unzip the directory `tar -xf deepnet.tar.gz`
- Go to the project directory `cd deepnet`
- Recreate environment from yml file `conda env create -f environment.yml`
- Activate the environment`conda activate deepnet-app`
- Run the app `cd deepnet` and `streamlit run deepnet.py`

#### Intro
The deepnet app is created using [Streamlit framework](https://www.streamlit.io/), which is still in early stages of development.
The framework is currently missing several key functions that will be hopefully added in the future and subsequently incorporated into the app.

So far, the application consists of two modules - data preprocessing, and training a neural net on that data. 
More functions will be added gradually (e.g. hyperparameter tuning or applying already trained model).

To select a task browse the select box in the side panel. 
For now, all input files (or folders) must be defined by an absolute path.

After running a task, there's several files exported to the defined output folder.
There's a log file with logged user input, warnings, etc. 
In the parameters.yaml file there are all the parameters set by a user. 
The yaml file can be imported in a future run to reproduce the set up.
For each task there is one common tsv table with one row per run.
You can then easily manage and compare results with different parameters setup.  

#### Data Preprocessing
In the first module, RNA or DNA sequence data are prepared to be fed into a neural network. 

You must select one or more **branches** corresponding to different ways of data manipulation.
* Raw sequence - select this branch if you wish to use pure sequence (one-hot encoded) as input. Requires reference genome in fasta file. 
* Conservation score - original sequence gets mapped against reference conservation files (the reference  files are required). 
* Secondary structure - secondary structure is estimated by [ViennaRNA](https://www.tbi.univie.ac.at/RNA/) package and the values are then one-hot encoded.
(Also requires the reference genome in fasta file).

All the samples fed to the neural network must be of the same length, which can be defined by **window size**.
Long sequences get shortened, while short sequences are filled in based on the reference by randomly placing window of selected size on the original sequence. 
Use **seed** parameter for reproducibility.

There can be an arbitrary number of **input coordinate files** in bed format, but you need at least two for the classification.
Each input file is handled as a separate class during model training (note that currently softmax activation function is used for all the models).
Selected class/es can be reduced to **target size** (again, use **seed** for reproducibility).
If you use this option in combination with dataset split by chromosomes, please make sure all classes in all the categories (train, test, etc.) 
contain at least some data, as some chromosomes might be lost when randomly reducing dataset size.

Supplied data must be split into training, validation and testing datasets.
This can be done either by choosing particular chromosomes per each category, or randomly, based on defined **ratio** (again, use **seed** for reproducibility).
If you choose to separate categories by chromosomes, fasta file with reference genome must be provided (the same one required for raw sequence and secondary structure branches).
List of available chromosomes is then inferred from the provided genome file (scaffolds are ignored) - that may take up to few minutes.
(Note: When selecting the chromosomes per category Streamlit will issue a warning 'Running read_and_cache(...).'. 
You may disregard that and continue selecting, filling in other attributes, or hit the run button to start processing the files.)

After all the parameters are set and selected, press the **run** button. The preprocessing might take several minutes to hours, 
depending on the amount of data, selected options, and hardware available.
If there's a mandatory field missing information or some input is incorrect, you will get a warning, and the app will not run.

Files with processed datasets are exported to the **output folder** defined at the beginning.  

#### Train a Model
Files provided within **input folder** are expected to be those exported by the Preprocess Data section of the deepnet app.
Selected **branches** must be the same as in the preprocessing step, **output folder** might also be the same.
You can select the checkbox to produce **Tensorboard files** (for more info see the [official site](https://www.tensorflow.org/tensorboard)).

In the second section you can pick a **batch size**, **number of epochs** to be trained, **optimizer** (SGD, Adam, or RMSprop), and **learning rate**.
When SGD is the optimizer of choice, you can use learning rate scheduler, instead of the usual fixed learning rate. 
**Metric** and **loss function** are predefined as accuracy and categorical crossentropy for now.

(Currently, learning rate finder and one cycle policy learning rate implementations are also being tested. 
When the lr finder is selected, a mock training is run for one epoch, and the resulting plot can be found in lr_finder.png file in the output folder.) 

Finally, the last section determines the network architecture. You must define architecture for each selected branch, and for the common part of the network after branches' concatenation.
First set a **number of layers** per each part. For each layer you must select its **type** (e.g. convolutional, dense, or LSTM).
If the checkbox is selected, you may set **advanced options** specific per layer type (e.g. dropout rate, or number of filters in a convolutional layer).

When all is set, press the **run** button to start training the model.
The training time depends on many variables (datasets size, network architecture, number of epochs, etc.).
You can monitor the progress on the chart indicating metric and loss function values.
Resulting model and all other files (e.g. tensorboard) are exported to the selected **output folder**. 

### Development
For now, if you wish to work with the app, test or develop the code, please contact me at Slack (@Eliska), and we can discuss the details.

#### How to contribute
Following the [shared repository model](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/about-collaborative-development-models),
the process is roughly as follows:
* Download the repository using `git clone git@gitlab.com:RBP_Bioinformatics/deepnet.git`
* Depending on the situation, create your own branch or use existing feature branch for development
    * Make sure you have the latest commits `git pull` (or `git fetch` first)
    * Use existing remote branch:
        * E.g. `git checkout -b streamlit_feature`
        * When finished, push your changes `git push` 
        (again, first make sure you have all the latest commits from remote branch)
    * Or create your own branch:
        * Checkout the branch you want to fork, e.g. `git checkout development`
        * Create and checkout the new branch, e.g. `git checkout -b your_name_development`
        * When making first push from this branch, set upstream tracking 
        `git push --set-upstream origin your_name_development`
* When you want to push a group of commits from a side branch to master or another protected branch, open a pull request
    * You might add a summary for the pull request, if it helps the clarity
    * Changes in the pull request can be reviewed and discussed by other collaborators
    * After that the pull request will be merged by an admin into the chosen base branch

#### To be done
See dedicated [Trello board](https://trello.com/b/me9e2k1e/rbp-binding) for a list of current tasks, issues, and additional information.

#### Tests 
(*Not yet available*)
