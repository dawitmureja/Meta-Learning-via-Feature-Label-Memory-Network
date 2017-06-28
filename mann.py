import numpy as np
import tensorflow as tf
import ipdb

bias = 1e-6
batch_size = 1

def slice_equally(x,size,n):
    return [x[:, i*(size): (i+1) * (size)] for i in range(n)]

def cosine_similarity(x,y):
    shape_y = y.get_shape().as_list()
    shape_x = x.get_shape().as_list()
    x_dot_y = tf.reshape(tf.matmul(x,tf.reshape(y,shape_y+[1])), shape_x[:-1])
    norm_x_y = tf.sqrt(tf.reshape((tf.reduce_sum(tf.multiply(y,y),1)), [-1,1]) * tf.reduce_sum(tf.multiply(x,x),2)) + bias
    return tf.div(x_dot_y,norm_x_y)

def clear_lu_memory(x,y):
    shape = x.get_shape().as_list()
    return tf.reshape(tf.reshape(x,[-1,shape[-1]]) * (1-tf.reshape(y,[-1,1])), shape)

def update_memory(x,y,z):
    shape_y = y.get_shape().as_list()
    shape_z = z.get_shape().as_list()
    y_mul_z = tf.matmul(tf.reshape(y,shape_y + [1]), tf.transpose(tf.reshape(z,shape_z + [1]), perm = [0,2,1]))
    return x + y_mul_z

def MANN(input_data, target_labels,memc_init,meml_init,batch_size,num_class,memory_shape, controller_size, input_size):
    wu_init = tf.Variable(tf.nn.softmax(tf.zeros([batch_size,memory_shape[0]], tf.float32, 'usage_weight')))
    ww_init = tf.Variable(tf.nn.softmax(tf.zeros([batch_size,memory_shape[0]], tf.float32, 'write_weight')))
    wr_init = tf.Variable(tf.nn.softmax(tf.zeros([batch_size,memory_shape[0]], tf.float32, 'read_weight')))
    r_init = tf.Variable(tf.zeros([batch_size, memory_shape[1]], name='read_vector'))
    h_init = tf.Variable(tf.zeros([batch_size, controller_size], name = 'hidden_unit'))
    c_init = tf.Variable(tf.zeros([batch_size, controller_size], name = 'cell_memory'))

    def glorot(shape):
        shape = np.array(shape)
        high = np.sqrt(6. / (np.sum(shape[:2]) * np.prod(shape[2:])))
        return tuple(shape),high


    with tf.variable_scope("Weights"):
        shape,high = glorot([controller_size,memory_shape[1]])
        W_key = tf.get_variable('W_key', shape = shape,initializer = tf.random_uniform_initializer(-1 * high , high))
        b_key = tf.get_variable('b_key', shape = [memory_shape[1]],initializer = tf.constant_initializer(0))
        shape, high = glorot([controller_size,1])
        W_sigma = tf.get_variable('W_sigma', shape = shape,initializer = tf.random_uniform_initializer(-1 * high,high))
        b_sigma = tf.get_variable('b_sigma', shape = [1],initializer = tf.constant_initializer(0))
        W_add = tf.get_variable('W_add',shape = [controller_size, memory_shape[1]],initializer = tf.random_normal_initializer(0,0.1))
        b_add = tf.get_variable('b_add',shape = [memory_shape[1]],initializer = tf.constant_initializer(0.1))
        W_addl = tf.get_variable('W_addl',shape = [controller_size, memory_shape[1]],initializer = tf.random_normal_initializer(0,0.1))
        b_addl = tf.get_variable('b_addl',shape = [memory_shape[1]],initializer = tf.constant_initializer(0.1))
        shape, high = glorot([input_size+ num_class,4 *controller_size])
        W_xh = tf.get_variable('W_xh', shape = shape, initializer = tf.random_uniform_initializer(-1 * high, high))
        b_h = tf.get_variable('b_h', shape = [4 * controller_size],initializer = tf.constant_initializer(0))
        shape,high = glorot([controller_size, 4 * controller_size])
        W_hh = tf.get_variable('W_hh', shape = shape,initializer = tf.random_uniform_initializer(-1 * high,high))
        shape, high = glorot([memory_shape[1], 4*controller_size])
        W_rh = tf.get_variable('W_rh', shape = shape ,initializer = tf.random_uniform_initializer(-1 * high, high))
        shape, high = glorot([memory_shape[1] + controller_size, num_class])
        W_out = tf.get_variable('W_out', shape = shape,initializer = tf.random_uniform_initializer(-1 * high,high))
        b_out = tf.get_variable('b_out', shape = [num_class],initializer = tf.constant_initializer(0))

    gamma = 0.95

    def at_each_step((M1_t0,M2_t0,h_t0,r_t0,c_t0,wr_t0,wu_t0,ww_t0), (x_t)):

        preactivations = tf.matmul(x_t,W_xh) + tf.matmul(r_t0,W_rh) + tf.matmul(h_t0,W_hh) + b_h
        gf_,gi_,go_,u_ = slice_equally(preactivations,controller_size,4)
        gf = tf.sigmoid(gf_)
        gi = tf.sigmoid(gi_)
        go = tf.sigmoid(go_)
        u = tf.tanh(u_)
        c_t = gf * c_t0 + gi * u
        h_t = go * tf.tanh(c_t)
        k_t = tf.tanh(tf.matmul(h_t,W_key) + b_key)
        sigma_t = tf.sigmoid(tf.matmul(h_t,W_sigma) + b_sigma)
        a_t = tf.tanh(tf.matmul(h_t,W_add) + b_add)
        al_t = tf.tanh(tf.matmul(h_t,W_addl) + b_addl)
        _,index = tf.nn.top_k(wu_t0,k = memory_shape[0])
        wlu_t  = tf.one_hot(index[:,-1], memory_shape[0])
        wwl_t = ww_t0
        ww_t = sigma_t * wr_t0 + (1-sigma_t) * wlu_t
        M1_t = clear_lu_memory(M1_t0,wlu_t)
        M1_t = update_memory(M1_t,ww_t,a_t)
        M2_t = update_memory(M2_t0, wwl_t,al_t)
        K_t = cosine_similarity(M1_t,k_t)
        wr_t = tf.nn.softmax(K_t)
        wu_t = gamma * wu_t0 + wr_t + ww_t
        r_t = tf.reshape(tf.matmul(tf.expand_dims(wr_t,1),M2_t),[batch_size,memory_shape[1]])
        return [M1_t,M2_t,h_t,r_t,c_t,wr_t, wu_t,ww_t]

    episode_length = int(target_labels.shape[1])
    output_shape = [batch_size * episode_length, num_class]
    one_hot_labels_flattened = tf.one_hot(tf.cast(tf.reshape(target_labels,[-1]), tf.int32), num_class)
    one_hot_labels = tf.reshape(one_hot_labels_flattened,(batch_size, episode_length,num_class))
    offset_target_labels =  tf.concat((tf.zeros_like(tf.expand_dims(one_hot_labels[:,0],1)),one_hot_labels[:,:-1]),axis = 1)
    input_data_mem = tf.concat([input_data, offset_target_labels], axis = 2)
    scanned_var = tf.scan(at_each_step, elems = tf.transpose(input_data_mem, perm = [1,0,2]), initializer = [memc_init,meml_init,h_init,r_init,c_init,wr_init,wu_init,ww_init])
    output = tf.transpose(tf.concat((scanned_var[2],scanned_var[3]),axis = 2), perm = [1,0,2])
    output_preactivation = tf.reshape(tf.matmul(tf.reshape(output,(episode_length*batch_size,-1)), W_out),[batch_size,episode_length,num_class]) + b_out
    predicted_labels_flattened = tf.nn.softmax(tf.reshape(output_preactivation,output_shape))
    predicted_labels = tf.reshape(predicted_labels_flattened,[batch_size,episode_length,num_class])
    parameters = [W_key,b_key,W_sigma,b_sigma,W_add, b_add,W_addl,b_addl,W_xh,b_h,W_hh,W_rh,W_out,b_out]
    return predicted_labels, parameters
