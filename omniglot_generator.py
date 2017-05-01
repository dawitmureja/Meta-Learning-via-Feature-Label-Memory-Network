import tensorflow as tf
import numpy as np
import random
import os
import ipdb
from scipy.misc import imread, imresize

def load_images(file_name, size):
    original_image = imread(file_name, flatten = True)
    resized_image = imresize(original_image, size = size).astype(original_image.dtype)
    return resized_image

def choose_samples(characters, labels, num_samples_per_class):
    images = []
    for i,character in zip(labels,characters):
        for image in random.sample(os.listdir(character), num_samples_per_class):
            images.append(tuple((i,os.path.join(character,image))))
    random.shuffle(images)
    return images

def generate_omniglot(data_folder, num_class, num_samples_per_class, image_size):
    # generate character folders
    character_folders = []
    for family in os.listdir(data_folder):
        if os.path.isdir(os.path.join(data_folder, family)):
            for character in os.listdir(os.path.join(data_folder,family)):
                character_folders.append(os.path.join(data_folder, family,character))


    #generate sampled image of (num_class,num_smaples_per_class)
    sampled_character_folders = random.sample(character_folders, num_class)
    choosen_samples = choose_samples(sampled_character_folders, np.arange(num_class),num_samples_per_class)

    #generate random sample of image for omniglot classification
    generated_samples = np.zeros((num_class * num_samples_per_class, np.prod(image_size)))
    generated_labels = np.zeros((num_class * num_samples_per_class,1))
    labels,images = zip(*choosen_samples)
    for i in range(generated_samples.shape[0]):
        generated_samples[i] = load_images(images[i], image_size).reshape(1,-1)
        generated_labels[i] = labels[i]
    return generated_samples, generated_labels.reshape([-1])

#a,b = generate_omniglot(data_folder = 'data/omniglot',num_class = 5 ,num_samples_per_class = 10,image_size = (28,28))
#ipdb.set_trace()
