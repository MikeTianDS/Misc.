import os
import gzip
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile

import logging

logger = logging.getLogger(__name__)


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(f):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth].

    Args:
      f: A file object that can be passed into a gzip reader.

    Returns:
      data: A 4D uint8 numpy array [index, y, x, depth].

    Raises:
      ValueError: If the bytestream does not start with 2051.

    """
    logger.info('Extracting' + f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError('Invalid magic number %d in MNIST image file: %s' %
                             (magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data.astype(np.float32)


def _dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot


def extract_labels(f, one_hot=False, num_classes=10):
    """Extract the labels into a 1D uint8 numpy array [index].

    Args:
      f: A file object that can be passed into a gzip reader.
      one_hot: Does one hot encoding for the result.
      num_classes: Number of classes for the one hot encoding.

    Returns:
      labels: a 1D uint8 numpy array.

    Raises:
      ValueError: If the bystream doesn't start with 2049.
    """
    logger.info('Extracting ' + f.name)
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError('Invalid magic number %d in MNIST label file: %s' %
                             (magic, f.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        if one_hot:
            return _dense_to_one_hot(labels, num_classes)
        return labels.astype(np.int32)


def get_data(data_dir='./data', one_hot=False):
    data_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
    with gfile.Open(data_path, 'rb') as f:
        train_images = extract_images(f)

    data_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
    with gfile.Open(data_path, 'rb') as f:
        train_labels = extract_labels(f, one_hot=one_hot)

    data_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
    with gfile.Open(data_path, 'rb') as f:
        test_images = extract_images(f)

    data_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')
    with gfile.Open(data_path, 'rb') as f:
        test_labels = extract_labels(f, one_hot=one_hot)

    logger.debug("train_images shape {}, train_labels shape {}".format(train_images.shape, train_labels.shape))
    logger.debug("test_images shape {}, test_labels shape {}".format(test_images.shape, test_labels.shape))

    return train_images, train_labels, test_images, test_labels


def get_dataset(train_size=1000, dev_size=100):
    train_images, train_labels, test_images, test_labels = get_data()

    ds_train = tf.data.Dataset.from_tensor_slices((train_images[:train_size], train_labels[:train_size]))

    ds_test = tf.data.Dataset.from_tensor_slices((test_images[:dev_size], test_labels[:dev_size]))

    ds_pred = tf.data.Dataset.from_tensor_slices(test_images[:dev_size])

    tf.logging.info("train dataset shape {}".format(ds_train.output_shapes))
    tf.logging.info("test dataset shape {}".format(ds_test.output_shapes))
    tf.logging.info("pred dataset shape {}".format(ds_pred.output_shapes))
    return ds_train, ds_test, ds_pred
