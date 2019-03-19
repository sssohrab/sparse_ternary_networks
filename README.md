# sparse_ternary_networks:
A package to compressievly represent vectorial data. This can be useful for many purposes, e.g., lossy compression where data fidelity is traded for compactness, fast similarity search where memory- and complexity-efficient search is preferred to the exact search, and also  as an efficient prior to regularizing ill-posed inverse-problems.


The package is an implementation of the ML-STC and ML-STC-Procrustean from [my PhD thesis](https://arxiv.org/abs/1901.08437). Note that the back-propagatable part, i.e., the extension to STNets is not included yet.

The network learns the weights of the multi-layer network and is supposed to provide, for each input vector (from the test set), a multi-layer ternary representation, i.e., $L$ layers of codes from $\{+1, 0, -1\}$ codes.

# Evaluation:
This is basically about the trade-offs between data fidelity, as measured by distortion, and compactness, as measured by rate (entropy). So for a fixed rate, which is specified by sparsity, code dimensions and the number of layers, the network should provide the best reconstruction quality possible.

## Installation:
Simply clone the package and add it into your Python 3 path.
The dependencies are very basic packages like NumPy and SciPy. Tested with Anaconda on mac and linux.

## The package:
`Networks.py` contains the 4 main classes of the package:
`fwdPass` runs a forward pass of the network on the test set. `BaseLearner` trains the network on a training set. `fwdPass_unit` and `BaseLearner` are the core units of the previous multi-layer classes, respectively.

`Tools.py` contains the basic functionality necessary for the classes of `Networks.py`.

`SearchTools.py` contains a set of non-optimized fast similarity search routines to be used on top of the other functionalities. Please note, however, that these search tools are currently highly non-optimized and are used only for demonstraion purposes.

## Examples:
The compression functionality and how to use the classes can be seen e.g. [here in this notebook.](https://github.com/sssohrab/DSW2018/blob/master/Correlated_sources.ipynb).

## To-Do's:
- To include the "refinement by backpropagation" functionality using PyTorch 1.0. 
- Examples and optimized codes for similarity search. This requires some C++ routines.
- Support for on-line scenarios using mini-batch versions of SVD.
