''' A simple MNIST download script

A collection of scripts to download the MNIST data in npy format from its
oiginal source: http://yann.lecun.com/exdb/mnist/

Files will always be downloaded and saved as ubyte first and will then be converted to npy. Data will always be saved in the mnist_data folder.

Note:
    Some of the ubyte conversion code is inspired by the script found here:
    http://pjreddie.com/projects/mnist-in-csv/
    
'''

#############
## Imports ##
#############

# Standard Library
import os
import gzip

try: # PY2
    from urllib import urlretrieve
except ImportError: # PY3
    from urllib.request import urlretrieve

# Other
import numpy as np


#################
## Fetch Mnist ##
#################

def fetch_mnist(redownload=False, verbose=True):
    ''' Get MNIST data in npy format

    Args:
        redownload=False (bool): force redownload, even if file already exists
    '''
    # check if data is already downloaded. If so, do not download again, except
    # when explicitly asked to do so
    if (os.path.exists('mnist_data/train.npy')
        and os.path.exists('mnist_data/test.npy')
        and not redownload):
        # load files from data folder
        return np.load('mnist_data/train.npy'), np.load('mnist_data/test.npy')

    # create folders
    if not os.path.isdir('mnist_data'):
        os.mkdir('mnist_data')

    # check if data is already downloaded. If so, do not download again, except
    # when explicitly asked to do so
    if not (os.path.exists('mnist_data/train_images.gz')
        and os.path.exists('mnist_data/train_labels.gz')
        and os.path.exists('mnist_data/test_images.gz')
        and os.path.exists('mnist_data/test_labels.gz')
        and not redownload):
        if verbose:
            print('downloading mnist data from http://yann.lecun.com/')
        # download data
        urlretrieve('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', 'mnist_data/train_images.gz')
        urlretrieve('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', 'mnist_data/train_labels.gz')
        urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', 'mnist_data/test_images.gz')
        urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', 'mnist_data/test_labels.gz')

    # fill numpy arrays:
    train = np.empty((60000,785), dtype='uint8')
    test = np.empty((10000,785), dtype='uint8')
    
    if verbose:
        print('converting .gz data to .npy')

    for type, npdata in [('train', train),('test', test)]:
        # open the files
        with gzip.open('mnist_data/%s_images.gz'%type, 'rb') as data,\
             gzip.open('mnist_data/%s_labels.gz'%type, 'rb') as labels:

            # skip the first bytes with metadata of the ubyte file:
            data.read(16)
            labels.read(8)

            # keep adding lines to the array until we reach the end
            # of the file (we break the loop when a TypeError is raised.)
            for i in range(npdata.shape[0]):
                npdata[i,0] = ord(labels.read(1))
                for j in range(784): # append the data after the label
                    npdata[i, j+1] = ord(data.read(1))

    np.save('mnist_data/train.npy', train)
    np.save('mnist_data/test.npy', test)
    
    if verbose:
        print('finished conversion.')

    return train, test
