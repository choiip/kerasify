import numpy as np
import struct

LAYER_DENSE = 1
LAYER_CONV_2D = 2
LAYER_FLATTEN = 3
LAYER_ELU = 4
LAYER_ACTIVATION = 5
LAYER_MAXPOOLING_2D = 6
LAYER_LSTM = 7
LAYER_EMBEDDING = 8
LAYER_BATCH_NORMALIZATION = 9

ACTIVATION_LINEAR = 1
ACTIVATION_RELU = 2
ACTIVATION_SOFTPLUS = 3
ACTIVATION_SIGMOID = 4
ACTIVATION_TANH = 5
ACTIVATION_HARD_SIGMOID = 6


def write_floats(f, floats):
    '''
    Writes floats to file in 1024 chunks.. prevents memory explosion
    writing very large arrays to disk when calling struct.pack().
    '''
    step = 1024
    written = 0

    for i in np.arange(0, len(floats), step):
        remaining = min(len(floats) - i, step)
        written += remaining
        f.write(struct.pack('=%sf' % remaining, *floats[i: i + remaining]))

    assert written == len(floats)


def export_activation(f, activation):
    if activation == 'linear':
        f.write(struct.pack('I', ACTIVATION_LINEAR))
    elif activation == 'relu':
        f.write(struct.pack('I', ACTIVATION_RELU))
    elif activation == 'softplus':
        f.write(struct.pack('I', ACTIVATION_SOFTPLUS))
    elif activation == 'tanh':
        f.write(struct.pack('I', ACTIVATION_TANH))
    elif activation == 'sigmoid':
        f.write(struct.pack('I', ACTIVATION_SIGMOID))
    elif activation == 'hard_sigmoid':
        f.write(struct.pack('I', ACTIVATION_HARD_SIGMOID))
    else:
        assert False, "Unsupported activation type: %s" % activation


def export_layer_dense(f, layer):
    weights = layer.get_weights()[0]
    biases = layer.get_weights()[1]
    activation = layer.get_config()['activation']

    f.write(struct.pack('I', LAYER_DENSE))
    f.write(struct.pack('I', weights.shape[0]))
    f.write(struct.pack('I', weights.shape[1]))
    f.write(struct.pack('I', biases.shape[0]))

    weights = weights.flatten()
    biases = biases.flatten()

    write_floats(f, weights)
    write_floats(f, biases)

    export_activation(f, activation)


def export_layer_conv2d(f, layer):
    # only border_mode=valid is implemented

    weights = layer.get_weights()[0]
    biases = layer.get_weights()[1]
    activation = layer.get_config()['activation']

    weights = weights.transpose(3, 0, 1, 2)
    # shape: (outputs, rows, cols, depth)

    f.write(struct.pack('I', LAYER_CONV_2D))
    f.write(struct.pack('I', weights.shape[0]))
    f.write(struct.pack('I', weights.shape[1]))
    f.write(struct.pack('I', weights.shape[2]))
    f.write(struct.pack('I', weights.shape[3]))
    f.write(struct.pack('I', biases.shape[0]))

    weights = weights.flatten()
    biases = biases.flatten()

    write_floats(f, weights)
    write_floats(f, biases)

    export_activation(f, activation)


def export_layer_maxpooling2d(f, layer):
    # only border_mode=valid is implemented

    pool_size = layer.get_config()['pool_size']

    f.write(struct.pack('I', LAYER_MAXPOOLING_2D))
    f.write(struct.pack('I', pool_size[0]))
    f.write(struct.pack('I', pool_size[1]))


def export_layer_lstm(f, layer):
    inner_activation = layer.get_config()['recurrent_activation']
    activation = layer.get_config()['activation']
    return_sequences = int(layer.get_config()['return_sequences'])

    weights = layer.get_weights()
    units = layer.units

    W_i = weights[0][:, :units]
    W_f = weights[0][:, units: units*2]
    W_c = weights[0][:, units*2: -units]
    W_o = weights[0][:, -units:]

    U_i = weights[1][:, :units]
    U_f = weights[1][:, units: units*2]
    U_c = weights[1][:, units*2: -units]
    U_o = weights[1][:, -units:]

    b_i = weights[2][:units]
    b_f = weights[2][units: units*2]
    b_c = weights[2][units*2: -units]
    b_o = weights[2][-units:]

    f.write(struct.pack('I', LAYER_LSTM))
    f.write(struct.pack('I', W_i.shape[0]))
    f.write(struct.pack('I', W_i.shape[1]))
    f.write(struct.pack('I', U_i.shape[0]))
    f.write(struct.pack('I', U_i.shape[1]))
    f.write(struct.pack('I', b_i.shape[0]))

    f.write(struct.pack('I', W_f.shape[0]))
    f.write(struct.pack('I', W_f.shape[1]))
    f.write(struct.pack('I', U_f.shape[0]))
    f.write(struct.pack('I', U_f.shape[1]))
    f.write(struct.pack('I', b_f.shape[0]))

    f.write(struct.pack('I', W_c.shape[0]))
    f.write(struct.pack('I', W_c.shape[1]))
    f.write(struct.pack('I', U_c.shape[0]))
    f.write(struct.pack('I', U_c.shape[1]))
    f.write(struct.pack('I', b_c.shape[0]))

    f.write(struct.pack('I', W_o.shape[0]))
    f.write(struct.pack('I', W_o.shape[1]))
    f.write(struct.pack('I', U_o.shape[0]))
    f.write(struct.pack('I', U_o.shape[1]))
    f.write(struct.pack('I', b_o.shape[0]))

    W_i = W_i.flatten()
    U_i = U_i.flatten()
    b_i = b_i.flatten()
    W_f = W_f.flatten()
    U_f = U_f.flatten()
    b_f = b_f.flatten()
    W_c = W_c.flatten()
    U_c = U_c.flatten()
    b_c = b_c.flatten()
    W_o = W_o.flatten()
    U_o = U_o.flatten()
    b_o = b_o.flatten()

    write_floats(f, W_i)
    write_floats(f, U_i)
    write_floats(f, b_i)
    write_floats(f, W_f)
    write_floats(f, U_f)
    write_floats(f, b_f)
    write_floats(f, W_c)
    write_floats(f, U_c)
    write_floats(f, b_c)
    write_floats(f, W_o)
    write_floats(f, U_o)
    write_floats(f, b_o)

    export_activation(f, inner_activation)
    export_activation(f, activation)
    f.write(struct.pack('I', return_sequences))


def export_layer_embedding(f, layer):
    weights = layer.get_weights()[0]

    f.write(struct.pack('I', LAYER_EMBEDDING))
    f.write(struct.pack('I', weights.shape[0]))
    f.write(struct.pack('I', weights.shape[1]))

    weights = weights.flatten()

    write_floats(f, weights)


def export_model(model, filename):
    with open(filename, 'wb') as f:
        model_layers = [
            l for l in model.layers if type(l).__name__ not in ['Dropout']]
        num_layers = len(model_layers)
        f.write(struct.pack('I', num_layers))

        for layer in model_layers:
            layer_type = type(layer).__name__

            if layer_type == 'Dense':
                export_layer_dense(f, layer)

            elif layer_type == 'Conv2D':
                export_layer_conv2d(f, layer)

            elif layer_type == 'Flatten':
                f.write(struct.pack('I', LAYER_FLATTEN))

            elif layer_type == 'ELU':
                f.write(struct.pack('I', LAYER_ELU))
                f.write(struct.pack('f', layer.alpha))

            elif layer_type == 'Activation':
                activation = layer.get_config()['activation']
                f.write(struct.pack('I', LAYER_ACTIVATION))
                export_activation(f, activation)

            elif layer_type == 'MaxPooling2D':
                export_layer_maxpooling2d(f, layer)

            elif layer_type == 'LSTM':
                export_layer_lstm(f, layer)

            elif layer_type == 'Embedding':
                export_layer_embedding(f, layer)

            elif layer_type == 'BatchNormalization':
                f.write(struct.pack('I', LAYER_BATCH_NORMALIZATION))
                f.write(struct.pack('f', layer.beta))
                f.write(struct.pack('f', layer.gamma))
                f.write(struct.pack('f', layer.epsilon))

            else:
                assert False, "Unsupported layer type: %s" % layer_type
