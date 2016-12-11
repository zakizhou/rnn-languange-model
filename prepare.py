from __future__ import division
from __future__ import absolute_import
from __future__ import absolute_import
import tensorflow as tf
import numpy as np
import sys


def prepare_corpus():
    pass


def arr2string(arr):
    return np.array(arr).tostring()


def convert_to_records(contexts, labels, sequence_lengths):
    writer = tf.python_io.TFRecordWriter("records/train.tfrecords")
    length = len(contexts)
    for i, (context, label, sequence_length) in enumerate(zip(contexts, labels, sequence_lengths)):
        sys.stdout.write("\r")
        sys.stdout.write("writing %d %% %d example to tfrecord file" % (i + 1, length))
        sys.stdout.flush()
        example = tf.train.Example(features=tf.train.Features(feature={
            'context': tf.train.Feature(bytes_list=[arr2string(context)]),
            'label': tf.train.Feature(bytes_list=[arr2string(label)]),
            'sequence_length': tf.train.Feature(int64_list=[sequence_length])
        }))
        writer.write(example.SerializeToString())
    writer.close()
