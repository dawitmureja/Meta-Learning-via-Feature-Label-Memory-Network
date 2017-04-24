import tensorflow as tf
import numpy as np
from omniglot_generator import generate_omniglot
from mann import MANN
import ipdb

data_folder = 'data/omniglot'
num_class = 5
num_samples_per_class = 10
memory_shape = (50,60)
controller_size = 100
image_size = (28,28)
input_size = 28 * 28
num_episodes = 1000

def test_mann():
    sess = tf.InteractiveSession()
    input_data = tf.placeholder(tf.float32,[num_class * num_samples_per_class, input_size])
    target_labels = tf.placeholder(tf.int32,[num_class * num_samples_per_class])

    predicted_labels, parameters = MANN(input_data, target_labels, num_class,memory_shape, controller_size, input_size)
    one_hot_target_labels = tf.one_hot(target_labels, num_class)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = predicted_labels, labels = one_hot_target_labels))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cost,var_list = parameters)
    correct_prediction = tf.equal(tf.argmax(predicted_labels,axis = 0), tf.argmax(target_labels, axis = 0))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    sess.run(tf.global_variables_initializer())
    for j in range(num_episodes):
        episode_input, episode_output = generate_omniglot(data_folder, num_class, num_samples_per_class, image_size)
        feed_dict = {input_data: episode_input,target_labels: episode_output}
        train_step.run(feed_dict)
        if (j % 10 == 0):
            train_accuracy = accuracy.eval(feed_dict)
            error = cost.eval(feed_dict)
            print("Episode: %d | Error: %g | Accuracy:%g " % (j, error, train_accuracy))
test_mann()



