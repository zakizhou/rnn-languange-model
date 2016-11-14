from tensorflow.python.ops import rnn, rnn_cell

NUM_UNITS = 40


def rnn(inputs):
    cell = rnn_cell.BasicLSTMCell(num_units=NUM_UNITS, state_is_tuple=True)
    outputs, state = rnn.rnn(cell, inputs)
    return outputs, state
