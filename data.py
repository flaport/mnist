''' MNIST Data Utils

A collection of scripts to download the MNIST data in different formats from its
oiginal source: http://yann.lecun.com/exdb/mnist/

Files will always be downloaded in ubyte first and then be converted to the
requested filetype.

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
from collections import OrderedDict

try: # PY2
    from urllib import urlretrieve
except ImportError: # PY3
    from urllib.request import urlretrieve

# Other
import numpy as np

#################################
## OrderedDict with attributes ##
#################################

class Dict(object):
    ''' Ordered Dictionary with attributes '''
    def __init__(self, *args, **kwargs):
        self.__dict__ = OrderedDict(*args, **kwargs)
    def __getitem__(self, key):
        return self.__dict__[key]
    def __setitem__(self, key, value):
        self.__dict__[key] = value
    def __getattr__(self, key):
        return self.__dict__[key]
    def __setattr__(self, key, value):
        if key == '__dict__':
            super(Dict, self).__setattr__('__dict__', value)
        else:
            self.__dict__[key] = value
    def values(self):
        return self.__dict__.values()
    def keys(self):
        return self.__dict__.keys()
    def items(self):
        return self.__dict__.items()
    def __repr__(self):
        return '{\n%s\n}'%'\n'.join('    %s : %s'%(repr(k), repr(v)) for k, v in self.__dict__.items())


######################
## Downloader Class #
######################

class DataRetriever():
    ''' Data Retriever Class '''
    def __init__(self, datadir='data'):
        ''' Set standard data library for MNIST data '''
        self.datadir = datadir.replace('\\','/')
        if self.datadir[-1] != '/':
            self.datadir = self.datadir + '/'

        ## GZ data locations (dictionary with attributes)
        self.gz = Dict(root=self.datadir)
        self.gz.dir = self.gz.root + 'gz/'
        self.gz.testdir = self.gz.dir + 'test/'
        self.gz.traindir = self.gz.dir + 'train/'
        self.gz.testlabels = self.gz.dir + 'test/labels.gz'
        self.gz.testdata = self.gz.dir + 'test/data.gz'
        self.gz.trainlabels = self.gz.dir + 'train/labels.gz'
        self.gz.traindata = self.gz.dir + 'train/data.gz'
        self.gz.exists = lambda : self.exists('gz')
        self.gz.createdirs = lambda : self.createdirs('gz')
        self.gz.download = self.gz_download

        ## CSV data locations
        self.csv = Dict(root=self.datadir)
        self.csv.dir = self.csv.root + 'csv/'
        self.csv.test = self.csv.dir + 'test.csv'
        self.csv.train = self.csv.dir + 'train.csv'
        self.csv.exists = lambda : self.exists('csv')
        self.csv.createdirs = lambda : self.createdirs('csv')
        self.csv.download = self.csv_download

        ## np data locations
        self.np = Dict(root=self.datadir)
        self.np.dir = self.np.root + 'np/'
        self.np.test = self.np.dir + 'test.npy'
        self.np.train = self.np.dir + 'train.npy'
        self.np.exists = lambda : self.exists('np')
        self.np.createdirs = lambda : self.createdirs('np')
        self.np.download = self.np_download
        self.np.get = self.np_get

        ## pandas
        self.pd = Dict()
        self.pd.get = self.pd_get

    def exists(self, datatype):
        ''' check if the data for the specific datatype has already been downloaded '''
        dic = getattr(self, datatype)
        for path in dic.values():
            if isinstance(path, str) and not os.path.exists(path):
                return False
        return True

    def createdirs(self, datatype):
        ''' create folders for the specific datatype if they do not already exist '''
        dic = getattr(self, datatype)
        if datatype == 'np':
            datatype = 'npy'
        for path in dic.values():
            if isinstance(path, str) and not os.path.exists(path) and not path.endswith('.' + datatype):
                os.mkdir(path)

    def gz_download(self, redownload=False):
        '''
        Download the MNIST data in its original form (ubyte zipped in gz form)
        from Yann Lecun's website

        Args:
            redownload=False (bool): force redownload, even if file already exists
        '''
        # check if data is already downloaded. If so, do not download again, except
        # when explicitly asked to do so
        if self.gz.exists() and not redownload:
            return # do nothing
        # create the necessary folders
        self.gz.createdirs()

        # download data
        urlretrieve('http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz', self.gz.traindata)
        urlretrieve('http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz', self.gz.trainlabels)
        urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz', self.gz.testdata)
        urlretrieve('http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz', self.gz.testlabels)

    def csv_download(self, redownload=False):
        ''' Download MNIST data in csv format

        Args:
            redownload=False (bool): force redownload, even if file already exists
        '''
        # check if data is already downloaded. If so, do not download again, except
        # when explicitly asked to do so
        if self.csv.exists() and not redownload:
            return # do nothing

        self.csv.createdirs()

        # download gz data:
        self.gz_download(redownload=redownload)


        for gzdir, csvdata in [(self.gz.traindir, self.csv.train),
                               (self.gz.testdir, self.csv.test)]:

            # open the three files
            with gzip.open(gzdir+'data.gz', 'rb') as data,\
                gzip.open(gzdir+'labels.gz', 'rb') as labels,\
                    open(csvdata, 'w') as outfile:

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

    def np_get(self, redownload=False):
        ''' Get MNIST data in npy format

        Args:
            redownload=False (bool): force redownload, even if file already exists
        '''
        # check if data is already downloaded. If so, do not download again, except
        # when explicitly asked to do so
        if self.np.exists() and not redownload:
            return np.load(self.np.train), np.load(self.np.test)

        # create folders
        self.np.createdirs()

        # download gz data:
        self.gz_download(redownload=redownload)

        # fill arrays:
        train = np.empty((60000,785), dtype='uint8')
        test = np.empty((10000,785), dtype='uint8')

        for gzdir, npdata in [(self.gz.traindir, train),
                               (self.gz.testdir, test)]:

            # open the three files
            with gzip.open(gzdir+'data.gz', 'rb') as data,\
                gzip.open(gzdir+'labels.gz', 'rb') as labels:

                # skip the first bytes with metadata of the ubyte file:
                data.read(16)
                labels.read(8)

                # keep adding lines to the csv file until we reach the end
                # of the file (we break the loop when a TypeError is raised.)
                for i in range(npdata.shape[0]):
                    npdata[i,0] = ord(labels.read(1))
                    for j in range(784): # append the data after the label
                        npdata[i, j+1] = ord(data.read(1))

        np.save(self.np.train, train)
        np.save(self.np.test, test)

        return train, test

    def np_download(self, redownload=False):
        ''' Download MNIST data in npy format

        Args:
            redownload=False (bool): force redownload, even if file already exists
        '''
        self.np.get(self, redownload=redownload)

    def pd_get(self, redownload=False):
        ''' Get MNIST data in a pandas dataframe

        Args:
            redownload=False (bool): force redownload, even if file already exists
        '''
        import pandas as pd

        # get numpy data
        np_train, np_test = self.np.get(redownload=redownload)

        pd_train = pd.DataFrame(
            data=np_train,
            index=np.arange(np_train.shape[0]),
            columns=['labels'] + ['pixel_%i'%i for i in range(784)],
        )

        pd_test = pd.DataFrame(
            data=np_test,
            index=np.arange(np_test.shape[0]),
            columns=['labels'] + ['pixel_%i'%i for i in range(784)],
        )
        return pd_train, pd_test


#############################
## Data Retriever instance ##
#############################

dataretriever = DataRetriever()
