import csv
import tensorflow as tf
import numpy as np


def csv_yield(csv_path):
    with open(csv_path, newline='') as f:
        next(f)  # skip the first line
        reader = csv.reader(f)
        for line in reader:
            feature, label = [float(i) for i in line[:-1]], int(line[-1])
            yield feature, label


def get_dataset(mode):
    if mode == 'train':
        return tf.data.Dataset.from_generator(
            lambda: csv_yield(r".\data\iris_training.csv"),
            output_types=(tf.float32, tf.int32))
    elif mode == 'eval':
        return tf.data.Dataset.from_generator(
            lambda: csv_yield(r".\data\iris_test.csv"),
            output_types=(tf.float32, tf.int32))
    else:
        data = [[5.1, 5.9, 6.9], [3.3, 3.0, 3.1], [1.7, 4.2, 5.4], [0.5, 1.5, 2.1]]
        data = np.array(data).T
        expected_y = ['Setosa', 'Versicolor', 'Virginica']
        return tf.data.Dataset.from_tensor_slices(data)


if __name__ == '__main__':

    ds = tf.data.Dataset.from_generator(
        lambda: csv_yield(r".\data\iris_training.csv"),
        output_types=(tf.float32, tf.int32)).batch(12)
    print(ds.output_shapes)

    ds_iter = ds.make_one_shot_iterator()

    # each column is a sample
    data = [[5.1, 5.9, 6.9],
            [3.3, 3.0, 3.1],
            [1.7, 4.2, 5.4],
            [0.5, 1.5, 2.1]]
    data = np.array(data).T
    ds2_iter = tf.data.Dataset.from_tensor_slices(data).batch(12).make_one_shot_iterator()

    with tf.Session() as sess:
        print(sess.run(ds_iter.get_next()))
        print(sess.run(ds2_iter.get_next()))


