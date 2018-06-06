
import tensorflow as tf


def create_rnn_cell(unit_type, num_units, num_layers, num_residual_layers, keep_prob, forget_bias=1.0):
    cell_list = []
    for i in range(num_layers):
        single_cell = _single_cell(
            unit_type=unit_type,
            num_units=num_units,
            forget_bias=forget_bias,
            keep_prob=keep_prob,
            residual_connection=(i >= num_layers - num_residual_layers))
        cell_list.append(single_cell)
    return tf.contrib.rnn.MultiRNNCell(cell_list)


def _single_cell(unit_type, num_units, forget_bias, keep_prob,
                 residual_connection=False, device_str=None, residual_fn=None):

    if unit_type == "lstm":
        single_cell = tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=forget_bias)
    elif unit_type == "gru":
        single_cell = tf.contrib.rnn.GRUCell(num_units)
    elif unit_type == "layer_norm_lstm":
        single_cell = tf.contrib.rnn.LayerNormBasicLSTMCell(num_units, forget_bias=forget_bias, layer_norm=True)
    elif unit_type == "nas":
        single_cell = tf.contrib.rnn.NASCell(num_units)
    else:
        raise ValueError("Unknown unit type %s!" % unit_type)

    single_cell = tf.contrib.rnn.DropoutWrapper(cell=single_cell, input_keep_prob=keep_prob)

    # Residual
    if residual_connection:
        single_cell = tf.contrib.rnn.ResidualWrapper(single_cell, residual_fn=residual_fn)

    # Device Wrapper
    if device_str:
        single_cell = tf.contrib.rnn.DeviceWrapper(single_cell, device_str)

    return single_cell
