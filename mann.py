import numpy as np
import tensorflow as tf

from omniglot_generator import generate_omniglot
import ipdb

controller_size = 200
def cosine_similarity(x,y):
    x_dot_y = tf.matmul(x,tf.reshape(y,[-1,1]))
    norm_x_y = tf.sqrt(tf.reduce_sum(tf.multiply(y,y)) * tf.reduce_sum(tf.multiply(x,x),1))
    z = tf.div(tf.reshape(x_dot_y,[-1]), tf.reshape(norm_x_y,[-1]))
    return z

def weights_and_biases(weight_shape):
    weight_initializer = tf.truncated_normal(shape = weight_shape, stddev = 0.1)
    bias_initializer = tf.constant(0.1, shape = [weight_shape[1]])
    return tf.Variable(weight_initializer), tf.Variable(bias_initializer)


def MANN(input_data, target_labels, num_class, memory_shape, controller_size, input_size):
    mem_init = tf.Variable(tf.zeros(memory_shape, tf.float32, 'memory'))
    wu_init = tf.Variable(tf.zeros([memory_shape[0],1], tf.float32, 'usage_weight'))
    wr_init = tf.Variable(tf.zeros([memory_shape[0],1], tf.float32, 'read_weight'))
    r_init = tf.Variable(tf.zeros([1, memory_shape[1]], name='read_vector'))

    W_key,b_key = weights_and_biases([controller_size, memory_shape[1]])
    W_sigma,b_sigma = weights_and_biases([controller_size, 1])
    W_out, b_out = weights_and_biases([memory_shape[1],num_class])
    gamma = 0.95

    W_fc1,b_fc1 = weights_and_biases([input_size + num_class,2048])
    W_fc2, b_fc2 = weights_and_biases([2048,1024])
    W_fc3,b_fc3 = weights_and_biases([1024,controller_size])



    def at_each_step((M_t0,r_t0,wr_t0,wu_t0), (x_t)):
        h_fc1 = tf.sigmoid(tf.matmul(tf.reshape(x_t, [1,-1]),W_fc1) + b_fc1)
        h_fc2 = tf.sigmoid(tf.matmul(h_fc1, W_fc2) + b_fc2)
        h_t= tf.sigmoid(tf.matmul(h_fc2,W_fc3) + b_fc3)
        k_t = tf.tanh(tf.matmul(h_t,W_key) + b_key)
        sigma_t = tf.tanh(tf.matmul(h_t,W_sigma) + b_sigma)
        wlu_t = np.zeros(wu_t0.get_shape().as_list())
        wlu_t[0] = 1
        _ ,index = tf.nn.top_k(tf.reshape(wu_t0, [-1]),k = wu_t0.get_shape().as_list()[0])
        wlu_t = tf.cast(wlu_t, tf.float32)
        wlu_t = tf.gather(wlu_t,index)
        ww_t = sigma_t * wr_t0 + (1-sigma_t) * wlu_t
        M_t0 = tf.multiply(M_t0, tf.reshape(1-wlu_t, [-1,1]))
        M_t = M_t0 + ww_t * k_t
        wr_t = tf.nn.softmax(cosine_similarity(M_t,k_t))
        wu_t = gamma * tf.reshape(wu_t0, [-1]) + wr_t + tf.reshape(ww_t,[-1])
        r_t = tf.matmul(tf.reshape(wr_t, [1,-1]),M_t)
        wr_t = tf.reshape(wr_t, [-1,1])
        wu_t = tf.reshape(wu_t, [-1,1])
        return [M_t, r_t, wr_t, wu_t]

    episode_length = target_labels.get_shape().as_list()[0]
    output_shape = (episode_length, num_class)
    one_hot_labels = tf.reshape(tf.one_hot(tf.cast(target_labels,tf.int32), num_class),[episode_length,num_class])
    offset_target_labels = tf.concat([one_hot_labels[:,1:], tf.reshape(tf.zeros(one_hot_labels.get_shape().as_list()[0]),[-1,1])],axis = 1)
    input_data_mem = tf.concat([input_data, offset_target_labels], axis = 1)
    scanned_var = tf.scan(at_each_step, elems = input_data_mem, initializer = [mem_init,r_init,wr_init,wu_init])
    predicted_labels = tf.matmul(tf.reshape(scanned_var[1],[episode_length,-1]),W_out) + b_out
    parameters = [W_key,b_key,W_sigma,b_sigma,W_fc1,b_fc1,W_fc2,b_fc2,W_fc3,b_fc3,W_out,b_out]
    return predicted_labels,parameters

