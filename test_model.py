import tensorflow as tf
import numpy as np
from omniglot_generator import generate_omniglot
from mann import MANN
import ipdb

data_folder = 'data/omniglot'
num_class = 5
num_samples_per_class = 10
memory_shape = (128,40)
controller_size = 200
image_size = (20,20)
input_size = 20 * 20
num_episodes = 150000

def trace_instance(target_labels):
    trace_counter = np.zeros((num_class,num_samples_per_class)).astype(np.float32)
    counter = [0] * num_class
    for i in range(num_class * num_samples_per_class):
        for j in range(num_class):
            if target_labels[i] == j:
                trace_counter[j][counter[j]] = i
                counter[j] += 1
    return trace_counter

def instance_accuracy2(predicted_labels, one_hot_target_labels):
    correct_prediction = tf.equal(tf.argmax(predicted_labels,axis = 1), tf.argmax(one_hot_target_labels, axis = 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy

def instance_accuracy_list(predicted_labels,target_labels,one_hot_target_labels, class_instance):
    accuracy_list = [0] * num_samples_per_class
    for k in range(num_samples_per_class):
        accuracy_list[k] = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.gather(predicted_labels,class_instance[:,k]),axis = 1),
                           tf.argmax(tf.gather(one_hot_target_labels, class_instance[:,k]), axis = 1)),tf.float32))
    return tf.cast(accuracy_list, tf.float32)



def test_mann():
    sess = tf.InteractiveSession()
    input_data = tf.placeholder(tf.float32,[num_class * num_samples_per_class, input_size])
    target_labels = tf.placeholder(tf.int32,[num_class * num_samples_per_class])
    mem_init = tf.placeholder(tf.float32,memory_shape)
    instance_list = tf.placeholder(tf.int32, [num_class,num_samples_per_class])

    predicted_labels, parameters = MANN(input_data, target_labels,mem_init, num_class,memory_shape, controller_size, input_size)
    with tf.variable_scope("Weights", reuse = True):
        W_key = tf.get_variable('W_key', shape = [controller_size, memory_shape[1]])
        b_key = tf.get_variable('b_key', shape = [memory_shape[1]])
        W_sigma = tf.get_variable('W_sigma', shape = [controller_size, 1])
        b_sigma = tf.get_variable('b_sigma', shape = [1])
        W_xh = tf.get_variable('W_xh', shape = [input_size + num_class, 4 * controller_size])
        b_h = tf.get_variable('b_h', shape = [4 * controller_size])
        W_hh = tf.get_variable('W_hh', shape = [controller_size, 4 * controller_size])
        W_rh = tf.get_variable('W_rh', shape = [ memory_shape[1], 4 * controller_size])
        W_out = tf.get_variable('W_out', shape = [memory_shape[1] + controller_size, num_class])
        b_out = tf.get_variable('b_out', shape = [num_class])

    parameters = [W_key,b_key,W_sigma,b_sigma,W_xh,b_h,W_hh,W_rh,W_out,b_out]
    one_hot_target_labels = tf.one_hot(target_labels, num_class)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = one_hot_target_labels, logits = predicted_labels), name = 'Cost')
    train_step = tf.train.AdamOptimizer(1e-3).minimize(cost,var_list = parameters)
    #insta = tf.cast(trace_instance(target_labels),tf.int32)
    accuracy_list = instance_accuracy_list(predicted_labels,target_labels,one_hot_target_labels,instance_list)

    tf.summary.scalar('Cost',cost)
    for i in range(num_samples_per_class):
        tf.summary.scalar('accuracy:'+str(i), accuracy_list[i])

    merge_all = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('/tmp/one_shot/tensorflow/')
    train_writer.add_graph(sess.graph)

    sess.run(tf.global_variables_initializer())
    for j in range(num_episodes):
        episode_input, episode_output = generate_omniglot(data_folder, num_class, num_samples_per_class, image_size)
        feed_dict = {input_data: episode_input,target_labels: episode_output, mem_init: np.zeros(memory_shape)}
        insta = trace_instance(episode_output)
        feed_dict2 = {input_data: episode_input,target_labels: episode_output,instance_list: insta, mem_init: np.zeros(memory_shape)}
        #if (j >= 0):
        train_step.run(feed_dict)
        error = cost.eval(feed_dict)
        #a = sess.run(accuracy_list.eval(feed_dict))
        accuracy = accuracy_list.eval(feed_dict2)
        summary = merge_all.eval(feed_dict2)
        train_writer.add_summary(summary,j)
            #class_instance = tf.cast(trace_instance(episode_output), tf.int32)
            #p = insta.eval(episode_output)

            #ipdb.set_trace()
            #k = class_instance
            #ipdb.set_trace()
            #instance_list2 = [0] * num_samples_per_class
            #for k in range(num_samples_per_class):
            #  instance_list2[k] = sess.run(instance_accuracy2(tf.gather(predicted_labels.eval(feed_dict),class_instance[:,k]), tf.gather(one_hot_target_labels.eval(feed_dict), class_instance[:,k])))
            #print("Episode: %d | Error: %g | " % (j+1, error)),
            #note that instance_accuracy is dependent on the num_of_samples_per_class
            #print("Instances:Accuracy => 1st:%g | 2nd:%g | 5th:%g |10th:%g " % (accuracy[0],accuracy[1],accuracy[4],accuracy[9]))
            #print("Instances:Accuracy => 1st:%g | 2nd:%g | 5th:%g |10th:%g " % (instance_list2[0],instance_list2[1],instance_list2[4],instance_list2[9]))
test_mann()

