# MNIST Digit Recognition

A Model Zoo applied to the MNIST digit recognition problem.

This repository containts a selection of possible ways to tackle the MNIST Digit Recognition problem with
different frameworks and with different models.

Some effort was made to make the implementation in different frameworks as similar as possible.


## Notebooks

### Linear Regression
Simple linear regression on the 784 pixels of the mnist digits.
* 00: [Linear Regression with sklearn](00_sklearn_linear_model.ipynb)

### Simple Neural Network
A simple fully connected feed forward neural network implementation on the 784 pixels of the mnist digits.
* 01: [Tensorflow](01_tensorflow_fcnn.ipynb)
* xy: Keras (todo)
* 02: [Pytorch](02_pytorch_fcnn.ipynb)

### Convolutional Neural Network
A convolutional neural network implementation on the 28x28 pixel mnist digit images.
* 03: [Tensorflow](02_tensorflow_cnn.ipynb)
* 04: [Keras](03_keras_cnn.ipynb)
* 05: [PyTorch](04_pytorch_cnn.ipynb)

### Recurrent Neural Network
A recurrent neural network (a single LSTM) recognizing the mnist digits by feeding each image pixel-by-pixel through
the network (pixel permuted sequential mnist problem).
* 06: [Tensorflow](05_tensorflow_rnn.ipynb)
* xy: Keras (todo)
* 07: [PyTorch](06_pytorch_rnn.ipynb)

### Efficient Unitary Neural Network
An EUNN recurrent unit recognizing the mnist digits by feeding each image pixel-by-pixel through the network (pixel permuted sequential mnist problem).
* xy: Tensorflow (todo)
* xy: Keras (todo)
* xy: PyTorch (todo)


## Other Content
A simple file to fetch the MNIST data from its original source ([http://yann.lecun.com/exdb/mnist](http://yann.lecun.com/exdb/mnist)) in a useful form:
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
