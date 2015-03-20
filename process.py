import numpy as np
import csv

class Data(object):
    def __init__(self,chr):
        self._chr = chr # chromosome number
        self._filename = "intersected_final_chr1_cutoff_20_"
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
        return data

    def read(self, detail = False):
        """
        Read all three data files for the given chromosome.
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