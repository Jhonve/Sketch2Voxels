import tensorflow as tf

def genConv2d(input_tensor, output_dim, kernel=[3, 3], strides=[1, 1], name="genConv2d"):
    with tf.variable_scope(name):
        conv = tf.contrib.layers.conv2d(input_tensor, output_dim, kernel, strides, padding="SAME", activation_fn=tf.nn.relu, 
                                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01), scope=name)
        return conv

def genDeconv2d(input_tensor, output_dim, kernel=[2, 2], strides=[2, 2], name="genDeconv2d"):
    # no activation function
    with tf.variable_scope(name):
        deconv = tf.contrib.layers.conv2d_transpose(input_tensor, output_dim, kernel, strides, padding="SAME", activation_fn=None, 
                                                weights_initializer=tf.truncated_normal_initializer(stddev=0.01), scope=name)
        return deconv

def disConv2d(input_tensor, output_dim, kernel=[3, 3], strides=[1, 1], name="disConv2d"):
    with tf.variable_scope(name):
        conv = tf.contrib.layers.conv2d(input_tensor, output_dim, kernel, strides, padding="SAME", activation_fn=tf.nn.leaky_relu, 
                                        weights_initializer=tf.truncated_normal_initializer(stddev=0.01), scope=name)
        return conv

def cropConcat(input_tensor_0 , input_tensor_1, concat_dim=1, name="concat"):
    with tf.variable_scope(name):
        t_0_shape = input_tensor_0.get_shape().as_list()
        t_1_shape = input_tensor_1.get_shape().as_list()

        if(t_0_shape[1] != t_1_shape[1] and t_0_shape[2] != t_1_shape[2]):
            offsets = [0, (t_0_shape[1] - t_1_shape[1]) // 2, (t_0_shape[2], t_1_shape[2]) // 2, 0]
            size = [-1, t_1_shape[1], t_1_shape[2], -1]
            t_0_crop = tf.slice(input_tensor_0, offsets, size)
            output = tf.concat([t_0_crop, input_tensor_1], concat_dim)
        else:
            output = tf.concat([input_tensor_0, input_tensor_1], concat_dim)
        return output

def l2Norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

def spectralNorm(w, iteration=1):
    w_shape = w.get_shape().as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])
    u = tf.get_variable("u", [1, w_shape[-1]], initializer = tf.truncated_normal_initializer(), trainable=False)
    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2Norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2Norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)
    return w_norm

def linear(input_tensor, output_size, sn=False, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
    shape = input_tensor.get_shape().as_list()
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        if sn:
            matrix = spectralNorm(matrix)
        biases = tf.get_variable("biases", [output_size], initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_tensor, matrix) + biases, matrix, biases
        else:
            return tf.matmul(input_tensor, matrix) + biases
