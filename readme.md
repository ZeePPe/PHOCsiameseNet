# Keyword Spotting based on N-gram retrieval

The repository contatins all code for the KWS spotting based on a core retrieval system that can recognaze sequences of few characters (N-grams)


# Preliminary Stages
## Install the conda environment
All this repo uses a conda environment. Firsto install the environment:
```
conda env create -f environment.yml
```
Once the env is installed, activate it:
```
conda activate pytorch
```

## Prepare the data folder
The `data` folder contains data for the different datasets.
The folder must contain a folder of the alphabet you want to use. An alphabet is nothing more than the set of all the basic symbols for spotting (In the case of N-grams, an alphabet is made up of all possible N-grams present). Each folder of an alphabet consists of subfolders, one for each symbol in the alphabet. Each symbol folder contains images of specimens belonging to that symbol.

![Example of an alphabet folder](img/alphabet_example.png)

The `data` folder must also contain a `rows` folder containing images of rows of handwritten documents to be used for searching. The folder is organized into subfolders, one for each handwritten document, each of which contains all the images of the document lines.

Finally, the `data` folder must also contain a `GT` folder containing the ground thruth files for the handwritten lines. The folder is organized into subfolders, one for each handwritten document, each of which contains all the text files with the transcripts of the text lines.

# The Important Stuff
## datasets
The `dataset` folder is a pyton package that contains all the modules to manage datasets.

## models
The `models` folder is a pyton package that contains all the modules to manage the Siamese network.


the package contains the `networks.py` module which provides various network structures
inside this module i implemented the `FrozenPHOCnet` class which is the same as the one we discussed
In practice, the network consists of a frozen PHOCnet and two fully connected towable layers.
The output of the network is a tensor of 256 elements

the package contains the `trainer.py` module which manages the training of a Siamese net.

## weights
The weights of the trained networks are saved in the `weights` folder


## Siamese network to measure N-gram similarity
The `siamese_measurer.py` module contains the `SiameseMeasurer` class. This class allows to calculate the distance between two images given an "arm" of a Siamese net
(Indeed, when the system saves the model of the network, it only saves one arm. In this way, memory space can be saved and the process of calculating the distances between several elements can be optimized)

## N-gram Retrieval System
The `ngram_retriever.py` module contains the `Spotter` class which allows you to search for an N-gram within a line of text. The search is performed by means of a sliding window and an instance of the SiameseMeasurer class

## Word Spotting
The `run_spotting.py` file allows you to perform word spotting using the sliding window based N-gram retrieval core defined in the `ngram_retriever.py` module.