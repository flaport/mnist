# MNIST Digit Recognition

A Model Zoo applied to the MNIST digit recognition problem.

This repository containts a selection of possible ways to tackle the MNIST Digit Recognition problem with
different frameworks and with different models.

Some effort was made to make the implementation in different frameworks as similar as possible.

## Notebooks
### Linear Regression
* 00: [Linear Regression with sklearn](00_sklearn_linear_model.ipynb)

### Simple Neural Network (todo)
* xy: Tensorflow (todo)
* xy: Keras (todo)
* xy: Pytorch (todo)

### Convolutional Neural Network
* 01: [Tensorflow](01_tensorflow_cnn.ipynb)
* 02: [Keras](02_keras_cnn.ipynb)
* 03: [Pytorch](03_pytorch_cnn.ipynb)

### Recurrent Neural Network
* 04: [Tensorflow](04_tensorflow_rnn.ipynb)
* xy: Keras (todo)
* xy: Pytorch (todo)

## Other Content
A Simple file to fetch the MNIST data from its original source ([http://yann.lecun.com/exdb/mnist](http://yann.lecun.com/exdb/mnist)) in a useful form:
* [fetch_mnist.py](fetch_mnist.py)

## Dependencies

### Python distribution
* [Anaconda Python 3.6](https://www.anaconda.com/download) or [Miniconda Python 3.6](https://conda.io/miniconda.html)

### General
* [`numpy`](http://www.numpy.org/): `conda install numpy`
* [`scipy`](http://www.scipy.org/): `conda install scipy`
* [`tqdm`](https://pypi.python.org/pypi/tqdm): `conda install tqdm` [progress bars]
* [`sklearn`](http://scikit-learn.org/): `conda install scikit-learn` [basic learning algorithms]
* [`matplotlib`](http://matplotlib.org/): `conda install matplotlib` [visualization]

### Deep learning  
* [`pytorch>=0.4.0`](http://pytorch.org/): `conda install pytorch -c pytorch`
* [`tensorflow>=1.2`](http://www.tensorflow.org/): `conda install tensorflow-gpu -c anaconda`
* [`keras`](http://keras.io/): `pip install keras`
