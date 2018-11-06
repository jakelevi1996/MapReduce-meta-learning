# MapReduce-meta-learning

This repository implements meta-learning algorithms from the literature such as Reptile ([Nichol, 2018](https://arxiv.org/abs/1803.02999)) as a map-reduce algorithm.

## File structure description

The file-structure of this repository is described by the list below (including some items which are in progress and not yet tracked), and was inspired by [this blog post by Morgan](https://blog.metaflow.fr/tensorflow-a-proposal-of-good-practices-for-files-folders-and-models-architecture-f23171501ae3) on [MetaFlow](https://blog.metaflow.fr/):

**Top level directory**:

- Read-me
- General `git` and IDE-related files
- `dockerfile`s and `shell`-scripts to automate their building and running
- `requirements.txt`
- **Code** folder:
  - `training.py`: where the magic happens
  - `inference.py`: for performing inference using pre-trained models
  - `main.py`: main script for running experiments, which involves specifying model-descriptions, training them, saving/restoring them, and then using them for inference
  - **data** folder:
    - `generate.py`: module for generating different types of synthetic data, EG for sinusoidal regression, regression on other types of signals, shape classification (both "What is inside/outside the shape" and "What kind of shape is this")
    module with functions for hard coded classification-rules for synthetic data, generating synthetic data, and loading synthetic data
    - `tasksets.py`: module containing class descriptions of TaskSet and DataSet objects, and methods for creating, saving and loading them
    - `preprocessing.py`: module for preprocessing datasets, EG MNIST
    - Data files for training and evaluation, in the `.npz` file format for storing multiple `Numpy` arrays, which can be loaded using the `tasksets` module
  - **models** folder:
    - `neuralmodels.py`: module containing a class for a generic neural network, a class for a reptile meta-learning model as a sub-class, including `adapt` and `meta_update` methods, and a class for a hrn model
    - Folders containing saved models, saved using [`tf.train.Saver.save()`](https://www.tensorflow.org/api_docs/python/tf/train/Saver#save)
  - **results** folder:
    - [Tensorboard](https://www.tensorflow.org/guide/summaries_and_tensorboard)-files saved using [`tf.summary.FileWriter.add_summary()`](https://www.tensorflow.org/api_docs/python/tf/summary/FileWriter#add_summary)

## Usage description

*Coming soon.*

## TODO

- Write full `readme.md` file
- Add docstrings to all functions
- Add noise to data generating function
- Investigate metatest extrapolation properties
- In `generate`, use `tf.data` API
- Add package structure to make imports work properly
- Add image plotting to sinusoid training, both before and after fast adaptation, using the `main` and `inference` modules
- Implement HRN training on MNIST
- Investigate using variance in VMP updates (isotropic at first, then diagonal; should first make notes on sequential Bayesian inference/VMP for Gaussian RVs)
