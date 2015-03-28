import numpy as np
import csv
from sklearn import preprocessing

class Data(object):
    def __init__(self,chr):
        self._chr = chr # chromosome number
        self._filename = "intersected_final_chr" + str(chr) + "_cutoff_20_"
        self.data_map = {} # Dictionary for all the data

    def _read_data(self,filename):
        """
        Read data file according to a given filename.
        :param filename: filename of the file to be read
        :return: file content in 2D list
        """
        f = open(filename)
        reader = csv.reader(f, delimiter='\t')
        data = list(reader)
        return np.asarray(data)

    def read(self, detail = False):
        """
        Read all three data files for the given chromosome. The data are given in strings.
        :param detail: If it is true, sizes of all the data are printed. False by default.
        :return: None
        """

        self.train = self._read_data(self._filename + "train_revised.bed")
        self.sample = self._read_data(self._filename + "sample.bed")
        self.test = self._read_data(self._filename + "test.bed")

        self.data_map["train"] = self.train
        self.data_map["sample"] = self.sample
        self.data_map["test"] = self.test

        if detail:
            self._data_detail("train")
            self._data_detail("sample")
            self._data_detail("test")

    def data_extract(self, strand_binary = False, strand_list = (0,1), pos_normalize = False, pos_range = (0,1)):
        """
        Extract starting position, ending position, strand, betas and chip numbers from
        the data file, converting to appropriate data types.
        :param strand_binary: Whether convert strand to binary types
        :param strand_list: Give a list of two numbers corresponding to - and +
        :param pos_normalize: Whether normalize the position
        :param pos_range: The range of normalized positions
        :return: None
        """
        # Extract starting position in integers
        self.train_start = self.train[:,1].astype(float)
        self.sample_start = self.sample[:,1].astype(float)
        self.test_start = self.test[:,1].astype(float)

        # Normalize starting position
        if pos_normalize:
            scaler = preprocessing.MinMaxScaler(feature_range=pos_range, copy=False)
            scaler.fit_transform(self.train_start)
            scaler.fit_transform(self.sample_start)
            scaler.fit_transform(self.test_start)

        # Extract ending position in integers
        self.train_end = self.train[:,2].astype(float)
        self.sample_end = self.sample[:,2].astype(float)
        self.test_end = self.test[:,2].astype(float)

        # Normalize starting position
        if pos_normalize:
            scaler = preprocessing.MinMaxScaler(feature_range=pos_range, copy=False)
            scaler.fit_transform(self.train_end)
            scaler.fit_transform(self.sample_end)
            scaler.fit_transform(self.test_end)

        # Extract strand information
        self.train_strand = self.train[:,3]
        self.sample_strand = self.sample[:,3]
        self.test_strand = self.test[:,3]
        # Convert strand information to binary numbers given in the strand_list
        if strand_binary:
            strand = {}
            strand["-"] = strand_list[0]
            strand["+"] = strand_list[1]
            self.train_strand = [strand[n] for n in self.train_strand]
            self.test_strand = [strand[n] for n in self.test_strand]
            self.sample_strand = [strand[n] for n in self.sample_strand]

        self.train_strand = np.array(self.train_strand)
        self.sample_strand = np.array(self.sample_strand)
        self.test_strand = np.array(self.test_strand)

        # Extract all betas
        self.train_beta = self.train[:,4:36].astype(float)
        self.sample_beta = self.sample[:,4].astype(float)
        self.test_beta = self.test[:,4].astype(float)

        # Extract the chip information
        self.train_chip = self.train[:,-1].astype(int)
        self.sample_chip = self.sample[:,-1].astype(int)
        self.test_chip = self.test[:,-1].astype(int)

        # Extract the indices of missing values in sample beta
        def findnan(beta_array):
            nan_ind = []
            nnan_ind = []
            for n in range(len(beta_array)):
                if np.isnan(beta_array[n]):
                    nan_ind.append(n)
                else:
                    nnan_ind.append(n)
            return np.array(nan_ind), np.array(nnan_ind)

        self.sample_nan, self.sample_non_nan = findnan(self.sample_beta)
        self.test_nan, self.test_non_nan = findnan(self.test_beta)

    def _data_detail(self, label):
        """
        Print out sizes of the data corresponding to the given label.
        :param label: The label for the data
        :return: None
        """
        print "For " + label + " data:"
        s = self.data_map[label]
        print "Number of rows: " + str(len(s))
        print "Number of cols: " + str(len(s[0])) + "\n"


def time_process(elapse_time):
    """
    Extract hour, minute and second from a given time in seconds
    :param elapse_time: time in seconds
    :return: hour, minute, second
    """
    from math import floor
    hour = floor(elapse_time / 3600)
    minute = floor((elapse_time - 3600*hour) / 60)
    second = elapse_time - 3600*hour - 60 * minute
    return hour, minute, second
