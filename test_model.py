import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import numpy as np
from Generator import OmniglotGenerator
from mann import MANN
import ipdb
import time
start = time.time()

train_data_folder = 'data/omniglot'
test_data_folder = 'test_data/omniglot'
num_test = 1000
batch_size = 5
num_class = 5
num_samples_per_class = 10
memory_shape = (64,40)
controller_size = 200
image_size = (20,20)
input_size = 20 * 20
num_episodes = 100000
test_accuracy = np.zeros(10).reshape(1,10)

def timer(time):
    hr = int(time)/3600
    minute = int(time - hr * 3600)/60
    sec = int(time - (hr * 3600) - (minute * 60))
    micro = time - (hr * 3600) - (minute * 60) - sec
    return hr, minute,sec,micro

def trace_instance(target_labels):
    trace_counter = np.zeros((num_class,num_samples_per_class)).astype(np.int32)
    counter = [0] * num_class
    for i in range(num_class * num_samples_per_class):
        for j in range(num_class):
            if target_labels[i] == j:
                trace_counter[j][counter[j]] = i
                counter[j] += 1
    return trace_counter

def instance_accuracy_list(predicted_labels,one_hot_target_labels, class_instance):
    accuracy_list = [0] * num_samples_per_class
    for k in range(num_samples_per_class):
        accuracy_list[k] = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.gather(predicted_labels,class_instance[:,k]),axis = 1),
                           tf.argmax(tf.gather(one_hot_target_labels, class_instance[:,k]), axis = 1)),tf.float32))
    return accuracy_list

def batch_instance(target_labels):
    batch_instance = [0] * batch_size
    for i in range(batch_size):
        batch_instance[i] = trace_instance(target_labels[i])
    return np.array(batch_instance)

def batch_accuracy(predicted_labels,one_hot_target_labels,class_instance):
    batch_accuracy = [0] *  batch_size
    for i in range(batch_size):
        batch_accuracy[i] = instance_accuracy_list(predicted_labels[i], one_hot_target_labels[i], class_instance[i])
    temp = tf.cast(batch_accuracy, tf.float32)
    return tf.reduce_mean(temp, axis = 0)

def test_mann():
    sess = tf.InteractiveSession()
    input_data = tf.placeholder(tf.float32,[batch_size,num_class * num_samples_per_class, input_size])
    target_labels = tf.placeholder(tf.int32,[batch_size,num_class * num_samples_per_class])
    memc_init = tf.placeholder(tf.float32,(batch_size,) + memory_shape)
    meml_init = tf.placeholder(tf.float32,(batch_size,) + memory_shape)
    instance_list = tf.placeholder(tf.int32, [batch_size,num_class,num_samples_per_class])

    train_generator = OmniglotGenerator(data_folder= train_data_folder, batch_size=batch_size, nb_samples=num_class, nb_samples_per_class=num_samples_per_class, max_rotation= -np.pi/16, max_shift=np.pi/16, max_iter=None)
    test_generator = OmniglotGenerator(data_folder= test_data_folder, batch_size=batch_size, nb_samples=num_class, nb_samples_per_class=num_samples_per_class, max_rotation=-10.0, max_shift=10.0, max_iter=None)
    predicted_labels,parameters = MANN(input_data, target_labels,memc_init,meml_init,batch_size,num_class,memory_shape, controller_size, input_size)
    one_hot_target_labels = tf.one_hot(target_labels, num_class)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = one_hot_target_labels, logits = predicted_labels), name = 'Cost')
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cost,var_list = parameters)

    accuracy_list = batch_accuracy(predicted_labels,one_hot_target_labels,instance_list)

    tf.summary.scalar('Cost',cost)
    for i in range(num_samples_per_class):
        tf.summary.scalar('accuracy:'+str(i), accuracy_list[i])

    merge_all = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('/tmp/tensorflow/')
    train_writer.add_graph(sess.graph)
    sess.run(tf.global_variables_initializer())
    for i, (episode_input, episode_output) in train_generator:
        if i >= num_episodes:
            break
        insta = batch_instance(episode_output)
        feed_dict = {input_data: episode_input,target_labels: episode_output, instance_list: insta,memc_init: np.zeros((batch_size,) + memory_shape), meml_init: np.zeros((batch_size,) + memory_shape)}
        train_step.run(feed_dict)
        summary = merge_all.eval(feed_dict)
        train_writer.add_summary(summary,i)
        if (i % 1000) == 0:
            episode_time = time.time()
            hr,minute,sec,micro = timer(episode_time - start)
            print("Episode %d time :%dhr: %dmin: %dsec: %.2fmicro_sec " % (i + 1,hr,minute,sec,micro))

    end = time.time()
    hr,minute,sec,micro = timer(end-start)
    print("Total training time :%dhr: %dmin: %dsec: %.2fmicro_sec " % (hr,minute,sec,micro))

    test_accuracy = np.zeros(10).reshape(1,10)
    for j, (test_input, test_output) in test_generator:
        if j >= num_test:
            break
        test_insta = batch_instance(test_output)
        test_feed_dict = {input_data: test_input, target_labels: test_output, instance_list: test_insta, memc_init: np.zeros((batch_size,) + memory_shape),meml_init: np.zeros((batch_size,) + memory_shape)}
        test_accuracy += accuracy_list.eval(test_feed_dict)
    for i in range(num_samples_per_class):
        if i == 0:
            print("test_accuracy: 1st instance: %.2f%%" % ((test_accuracy[0][i] * 100)/num_test))
        elif i == 1:
            print("test_accuracy: 2nd instance: %.2f%%" % ((test_accuracy[0][i] * 100)/num_test))
        elif i == 2:
            print("test_accuracy: 3rd instance: %.2f%%" % ((test_accuracy[0][i] * 100)/num_test))
        else:
            print("test_accuracy: %dth instance: %.2f%%" % (i+1,(test_accuracy[0][i] * 100)/num_test))
test_mann()
