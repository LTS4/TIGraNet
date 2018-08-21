# PyTorch implementation of TIGraNet

This project implements the TIGranet model proposed in ["Graph-based Isometry Invariant Representation Learning", R. Khasanova and P. Frossard (ICML-2017)](http://proceedings.mlr.press/v70/khasanova17a.html) using the PyTorch framework.

## Project structure

The project environment setup is composed of the following parts:

Some folders:
  * `saved_data`:  data used from Theano after preprocessing with corresponding pretrained weights.
  * `data`: raw data for each dataset used in the experiment ([mnist](http://yann.lecun.com/exdb/mnist/), [eth80](http://datasets.d2.mpi-inf.mpg.de/eth80/eth80-cropped256.tgz)).
  * `saved_models`: models saved during training.
  * `figures`: plots of performance and some comparison figures.
  * `debug_mnist_012`: files with all intermediary steps saved individually for mnist_012 dataset (from both PyTorch and Theano frameworks).
  * `debug_mnist_rot`: files with all intermediary steps saved individually for mnist_rot dataset (from both PyTorch and Theano frameworks).
  * `debug_mnist_eth80`: files with all intermediary steps saved individually for eth80 dataset (from both PyTorch and Theano frameworks).

And several python modules about:
  * the datasets:
    * **`datasets.py`**: loads the raw datasets with the specific transformations and splits.
    * **`saved_datasets.py`**: loads the saved datasets from the Theano implementation.
    * **`custom_transform.py`**: custom tranformations used for preprocessing on images.
    * **`loader.py`**: custom loader for mnist_012 dataset.
  * the models:
    * **`models.py`**: model description for each dataset (mnist_012, mnist_rot, mnist_trans, eth80)
    * **`layers.py`**: layer description for each new layer (spectral convolutional, dynamic pooling and statistical)
    * **`graph.py`**: main functions applied on the graph (Laplacian, normalized Laplacian, Chebyshev polynomials, ...)
  * the debugging process:
    * **`comparison_debug.py`**: comparison of intermediary steps between PyTorch and Theano implementations.
    * **`layers_debug.py`**: debugging module for layers.
  * the training and testing:
    * **`train.py`**: training module for the models.
    * **`evaluate.py`**: inference module for testing set.
  * the analysis tools:
    * **`plot.py`**: plot some metrics like the loss and % of errors.
  * the configurations:
    * **`configurations.py`**: configuration values for the project (batch size, learning rate, ...)
    * **`paths.py`**: different paths for the project.
  * the utilitary functions:
    * **`utils.py`**: auxiliary functions (train/valid/test split, save/load models, ...)

In addition, all functions come with a oneline description to better understand its purpose.

## Instructions to run the code

In order to train or evaluate a specific dataset, the following command line is expected.

```terminal
>> python3 train.py mnist_012
```

```terminal
>> python3 evaluate.py mnist_012
```

Notice that only one additional argument is allowed. The available datasets you can enter are: `mnist_012`, `mnist_rot`, `mnist_trans`, `eth80`. If you don't enter the command correctly a message will appear with some instructions and/or information.

The remaining modules must be called without additional argument.

## Instructions to install required libraries



### Conda

Anaconda is used as the package manager. Thus, all needed libraries are installed through the `conda` command line. If you don't have it installed, you can easily follow the steps from their [download](https://www.anaconda.com/download/) page. The version with `Python 3.6` is recommended.

### PyTorch

Once Conda is installed, we can proceed with the installation of [PyTorch](https://pytorch.org/). Run the following command.

```terminal
>> conda install pytorch torchvision -c pytorch
```

### tqdm

Finally, we install a smart progress meter. As it does not come pre-installed by default, we can simply install it with following command line:

```terminal
>> conda install tqdm
```

You are now ready to run the code!

---
**NOTE**

Code is runnable with the following configuration:

- conda 4.5.4
- conda-built 3.10.5
- python 3.6.5
- pytorch 0.4.0
- torchvision 0.2.1
- tqdm 4.23.4

---

## CUDA support

The code can run on CUDA flawlessly without any particular modification in the code.

## Authors:
This code was implemented by Mateus Fonseca Joel <joel.fonseca@epfl.ch> and it is PyTorch re-implementation version of the original code from the paper.
