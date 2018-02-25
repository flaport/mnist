'''
Download MNIST DataSet from its original source:
http://yann.lecun.com/exdb/mnist/

And convert the quite unpractical ubyte file to csv.

This code is an adapted version from the script found here:
http://pjreddie.com/projects/mnist-in-csv/
'''

#############
## Imports ##
#############

import gzip

try: # PY2
    from urllib import urlretrieve
except ImportError: # PY3
    from urllib.request import urlretrieve


##############
## Download ##
##############

def download():
    ''' Download the MNIST data in its original form from Yann Lecun's website'''
    train_images_tmp_file, _ = urlretrieve('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz')
    train_labels_tmp_file, _ = urlretrieve('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz')
    test_images_tmp_file, _  = urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz')
    test_labels_tmp_file, _  = urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz')
    return train_images_tmp_file, train_labels_tmp_file, test_images_tmp_file, test_labels_tmp_file


#############
## Convert ##
#############

def convert(filename, data, labels):
    ''' convert
    This function converts the data and labels file (in gz format) to
    a csv file.

    Arguments
    ---------
    filename (str): filename of the resulting csv file
    data (str): filename of the data file (gz file)
    labels (str): filename of the labels file (gz file)
    '''
    # open the three files
    with gzip.open(data, 'rb') as data,\
         gzip.open(labels, 'rb') as labels,\
              open(filename, 'w') as outfile:

        # skip the first bytes with metadata of the ubyte file:
        data.read(16)
        labels.read(8)

        # keep adding lines to the csv file until we reach the end
        # of the file (we break the loop when a TypeError is raised.)
        newline = ''
        while True:
            image = [None]*785
            try: # try to read the next byte in the labels file
                image[0] = str(ord(labels.read(1)))
            except TypeError: # break the loop if there are no labels left
                break
            for j in range(784): # append the data after the label
                image[j+1] = str(ord(data.read(1)))

            # Write this image (label + data) to a single line in the csv file:
            outfile.write(newline+','.join(image))
            newline = '\n'


###################
## Retrieve Data ##
###################

if __name__ == '__main__':
    (train_images_tmp_file,
    train_labels_tmp_file,
    test_images_tmp_file,
    test_labels_tmp_file) = download()

    convert(
        'mnist_train.csv',
        train_images_tmp_file,
        train_labels_tmp_file,
    )

    convert(
        'mnist_test.csv',
        test_images_tmp_file,
        test_labels_tmp_file,
    )
