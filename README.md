# MapReduce-meta-learning

This repository implements meta-learning algorithms from the literature such as Reptile ([Nichol, 2018](https://arxiv.org/abs/1803.02999)) as a map-reduce algorithm.

## File Structure

*Coming soon.*

## Usage description

*Coming soon.*

## TODO

- Write full `readme.md` file
- Add docstrings to all functions
- Add noise to data generating function
- Investigate metatest extrapolation properties
- In `generate_data`, each task should be further partitioned into training and test sets
- In `generate_data`, use `tf.data` API
- Maybe in `generate_data`, each task-set such be an object, based on a `TaskSet` class, helping to improve code reuse ???
