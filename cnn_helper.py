import tensorflow as tf


def layer_dropout(inputs, residual, keep_prob):
    pred = tf.random_uniform([]) < keep_prob
    outputs = tf.cond(pred, lambda: inputs, lambda: residual)
    return outputs


def conv(inputs, num_layers, kernel_size, filter_size, keep_prob, scope):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        for i in range(num_layers):
            scope = 'layer.{}'.format(i)
            inputs = tf.contrib.layers.layer_norm(inputs)
            outputs = depthwise_separable_convolution(inputs, kernel_size, filter_size, scope)
            if i > 0:
                inputs = layer_dropout(outputs, inputs, keep_prob)
            else:
                inputs = outputs
        return outputs


def depthwise_separable_convolution(inputs, kernel_size, num_filters, scope="depthwise_separable_convolution"):
    with tf.variable_scope(scope):
        inputs = tf.expand_dims(inputs, 1)
        input_depth = inputs.get_shape()[-1]
        depthwise_filter = tf.get_variable("depthwise_filter", [1, kernel_size, input_depth, 1], dtype=tf.float32)
        pointwise_filter = tf.get_variable("pointwise_filter", [1, 1, input_depth, num_filters], dtype=tf.float32)
        outputs = tf.nn.separable_conv2d(inputs,
                                        depthwise_filter,
                                        pointwise_filter,
                                        strides=[1, 1, 1, 1],
                                        padding="SAME")
        bias = tf.get_variable("bias", outputs.shape[-1])
        outputs += bias
        outputs = tf.nn.tanh(outputs)
        return tf.squeeze(outputs, 1)