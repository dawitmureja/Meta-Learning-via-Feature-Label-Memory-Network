import numpy as np
from scipy.misc import toimage, imsave,imresize,imshow
import ipdb
import random

#images = np.loadtxt('ex3data1.csv',delimiter = ',')
def visualize_data(data,sample_width = None, resize = False):
    if sample_width is None:
        sample_width = int(round(np.sqrt(data.shape[1])))

    rows,cols = data.shape
    sample_height = int(cols / sample_width)
    display_rows = np.floor(np.sqrt(rows)).astype(int)
    display_cols = np.ceil(rows/display_rows).astype(int)
    padding = 1
    #you can multiply diplay_array by -1
    display_array = np.ones((((padding + display_rows) * (padding + sample_height) - sample_height), ((padding + display_cols) * (padding + sample_width)-sample_width)))

    current_sample = 0
    for j in range(display_rows):
        for i in range(display_cols):
            if current_sample > rows: break
            max_value = np.max(np.abs(data[current_sample]))
            display_array[padding + i * (padding + sample_height) : (i +padding) * (padding + sample_height),
                          padding + j * (padding + sample_width) :  (j + padding) * (padding + sample_width)] = data[current_sample].reshape(
                              sample_width, sample_height)#/max_value
            current_sample += 1
        if current_sample > rows:break
    if resize:
        display_array = imresize(display_array,(200,200))

    return display_array.T
def display_data_sample(data,num_samples,sample_width = None,resize = False):
    index = np.arange(data.shape[0])
    random.shuffle(index)
    sample = np.zeros((num_samples, data.shape[1]))
    for i in range(num_samples):
        sample[i] = data[index[i]]
    return visualize_data(sample,sample_width,resize = resize)
#a = display_data_sample(images,40,resize = True)
#imshow(a)

