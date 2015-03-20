import numpy as np
import csv

filename_train = './data/intersected_final_chr1_cutoff_20_train_revised.bed'

dat = []

with open(filename_train, 'rb') as f:
    reader = csv.reader(f, delimiter = '\t')
    for row in reader:
        dat.append(row)

print dat[1]