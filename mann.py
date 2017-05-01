import numpy as np
import tensorflow as tf

from omniglot_generator import generate_omniglot
import ipdb

bias = 1e-6

def slice_equally(x,size,n):
    return [x[:, i*(size): (i+1) * (size)] for i in range(n)]

def cosine_similarity(x,y):
    x_dot_y = tf.matmul(x,tf.reshape(y,[-1,1]))
    norm_x_y = tf.sqrt((tf.reduce_sum(tf.multiply(y,y)) * tf.reduce_sum(tf.multiply(x,x),1)) + bias)
    z = tf.div(tf.reshape(x_dot_y,[-1]), tf.reshape(norm_x_y,[-1]))
    return z

#def weights_and_biases(weight_shape):
    #weight_initializer = tf.truncated_normal(shape = weight_shape, stddev = 0.1)
    #bias_initializer = tf.constant(0.1, shape = [weight_shape[1]])
    #return tf.Variable(weight_initializer), tf.Variable(bias_initializer)


def MANN(input_data, target_labels,mem_init,num_class, memory_shape, controller_size, input_size):
    #mem_init = tf.Variable(tf.zeros(memory_shape, tf.float32, 'memory'))
    wu_init = tf.Variable(tf.reshape(tf.nn.softmax(tf.reshape(tf.zeros([memory_shape[0],1], tf.float32, 'usage_weight'),[-1])),[-1,1]))
    wr_init = tf.Variable(tf.reshape(tf.nn.softmax(tf.reshape(tf.zeros([memory_shape[0],1], tf.float32, 'read_weight'),[-1])),[-1,1]))
    r_init = tf.Variable(tf.zeros([1, memory_shape[1]], name='read_vector'))
    h_init = tf.Variable(tf.zeros([1, controller_size], name = 'hidden_unit'))
    c_init = tf.Variable(tf.zeros([1, controller_size], name = 'cell_memory'))


    with tf.variable_scope("Weights"):
        W_key = tf.get_variable('W_key', shape = [controller_size, memory_shape[1]],initializer = tf.random_normal_initializer(0,0.1))
        b_key = tf.get_variable('b_key', shape = [memory_shape[1]],initializer = tf.constant_initializer(0.1))
        W_sigma = tf.get_variable('W_sigma', shape = [controller_size, 1],initializer = tf.random_normal_initializer(0,0.1))
        b_sigma = tf.get_variable('b_sigma', shape = [1],initializer = tf.constant_initializer(0.1))
        W_xh = tf.get_variable('W_xh', shape = [input_size + num_class, 4 * controller_size],initializer = tf.random_normal_initializer(0,0.1))
        b_h = tf.get_variable('b_h', shape = [4 * controller_size],initializer = tf.constant_initializer(0.1))
        W_hh = tf.get_variable('W_hh', shape = [controller_size, 4 * controller_size],initializer = tf.random_normal_initializer(0,0.1))
        W_rh = tf.get_variable('W_rh', shape = [ memory_shape[1], 4 * controller_size],initializer = tf.random_normal_initializer(0,0.1))
        W_out = tf.get_variable('W_out', shape = [memory_shape[1] + controller_size, num_class],initializer = tf.random_normal_initializer(0,0.1))
        b_out = tf.get_variable('b_out', shape = [num_class],initializer = tf.constant_initializer(0.1))
    #W_key,b_key = weights_and_biases([controller_size, memory_shape[1]])
    #W_add,b_add = weights_and_biases([
    #W_sigma,b_sigma = weights_and_biases([controller_size, 1])
    #W_xh,b_h = weights_and_biases([input_size + num_class,  4 * controller_size])
    #W_hh,_ = weights_and_biases([controller_size, 4 * controller_size])
    #W_rh,_ = weights_and_biases([memory_shape[1], 4 * controller_size])
    #W_out, b_out = weights_and_biases([memory_shape[1] + controller_size,num_class])

    gamma = 0.95

    #feed_forward network
    #W_fc1,b_fc1 = weights_and_biases([input_size + num_class,1024])
    #W_fc2, b_fc2 = weights_and_biases([1024,512])
    #W_fc3,b_fc3 = weights_and_biases([512,controller_size])



    def at_each_step((M_t0,r_t0,h_t0,c_t0,wr_t0,wu_t0), (x_t)):

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

        preactivations = tf.matmul(tf.reshape(x_t, [1,-1]),W_xh) + tf.matmul(r_t0,W_rh) + tf.matmul(h_t0,W_hh) + b_h
        gf_,gi_,go_,u_ = slice_equally(preactivations,controller_size,4)
        gf = tf.sigmoid(gf_)
        gi = tf.sigmoid(gi_)
        go = tf.sigmoid(go_)
        u = tf.tanh(u_)
        c_t = gf * c_t0 + gi * u
        h_t = go * tf.tanh(c_t)
        #h_fc1 = tf.sigmoid(tf.matmul(tf.reshape(x_t, [1,-1]),W_fc1) + b_fc1)
        #h_fc2 = tf.sigmoid(tf.matmul(h_fc1, W_fc2) + b_fc2)
        #h_t= tf.sigmoid(tf.matmul(h_fc2,W_fc3) + b_fc3)
        k_t = tf.tanh(tf.matmul(h_t,W_key) + b_key)
        sigma_t = tf.tanh(tf.matmul(h_t,W_sigma) + b_sigma)
        #wlu_t = np.zeros(wu_t0.get_shape().as_list())
        #wlu_t[-1] = 1
        _ ,index = tf.nn.top_k(tf.reshape(wu_t0, [-1]),k = memory_shape[0])
        wlu_t = tf.reshape(tf.one_hot(index[-1],memory_shape[0]),[-1,1])
        #wlu_t = tf.gather(wlu_t,index)
        ww_t = sigma_t * wr_t0 + (1-sigma_t) * wlu_t
        M_t0 = tf.multiply(M_t0, tf.reshape(1-wlu_t, [-1,1]))
        M_t = M_t0 + ww_t * k_t
        wr_t = tf.reshape(tf.nn.softmax(cosine_similarity(M_t,k_t)),[-1,1])
        wu_t = gamma * wu_t0 + wr_t + ww_t
        r_t = tf.matmul(tf.reshape(wr_t, [1,-1]),M_t)
        return [M_t, r_t,h_t,c_t,wr_t, wu_t]

    episode_length = int(target_labels.shape[0])
    one_hot_labels = tf.reshape(tf.one_hot(tf.cast(target_labels,tf.int32), num_class),[episode_length,num_class])
    offset_target_labels = tf.concat((tf.reshape(tf.zeros(one_hot_labels.get_shape().as_list()[1]), [1,-1]),one_hot_labels[:-1,:]),axis = 0)
    input_data_mem = tf.concat([input_data, offset_target_labels], axis = 1)
    scanned_var = tf.scan(at_each_step, elems = input_data_mem, initializer = [mem_init,r_init,h_init,c_init,wr_init,wu_init])
    preactivation = tf.concat((tf.reshape(scanned_var[2], [episode_length,-1]), tf.reshape(scanned_var[1],[episode_length,-1])),axis = 1)
    predicted_labels = tf.nn.softmax(tf.matmul(preactivation,W_out) + b_out)
    parameters = [W_key,b_key,W_sigma,b_sigma,W_xh,b_h,W_hh,W_rh,W_out,b_out]
    return predicted_labels,parameters

#a,b = generate_omniglot(data_folder = 'data/omniglot',num_class = 3 ,num_samples_per_class = 5,image_size = (20,20))
#c,d = MANN(a,b,3,(20,20),40,400)
#ipdb.set_trace()
#num_class = 5
#num_samples_per_class = 10
#trace_counter = np.empty((num_class, num_samples_per_class)).astype(np.int32)
#counter = [0] * num_class
#for i in range(b.shape[0]):
 #   for j in range(num_class):
  #      if b[i] == j:
   #             trace_counter[j][counter[j]] = i
    #            counter[j] += 1
#ipdb.set_trace()

