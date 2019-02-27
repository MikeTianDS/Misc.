import numpy as np
import tensorflow as tf
from tensorflow_template.base.base_model import BaseModel, Config
from examples.mnist.data_helper import get_data, get_dataset

import logging

logging.basicConfig(format='[%(name)s] : %(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)


class MnistModel(BaseModel):
    """"""

    def _init_graph(self):
        """"""
        self.features = tf.placeholder(tf.float32, [None, 28, 28, 1], name='features')
        self.labels = tf.placeholder(tf.int32, [None], name='labels')
        self.training = tf.placeholder(tf.bool, (), name='training')

        # For tf newer, the most important is to know the in/out shape of each layer
        # Input size:  [batch_size, 28, 28, 1]
        # Output size: [batch_size, 28, 28, 32]
        # The strides is (1, 1) default, so it will not change
        # the size of image which is (28, 28) when `padding="same"`
        conv1 = tf.layers.conv2d(inputs=self.features,
                                 filters=32,
                                 kernel_size=[5, 5],
                                 padding="same",
                                 activation=tf.nn.relu)

        # In:  [batch_size, 28, 28, 32]
        # Out: [batch_size, 14, 14, 32], here 14 = 28/strides = 28/2 = 14
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

        # In:  [batch_size, 14, 14, 32]
        # Out: [batch_size, 14, 14, 64]
        conv2 = tf.layers.conv2d(inputs=pool1,
                                 filters=64,
                                 kernel_size=[5, 5],
                                 padding="same",
                                 activation=tf.nn.relu)

        # In:  [batch_size, 14, 14, 64]
        # Out: [batch_size, 7, 7, 64], where 7 is computed same as pool1
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

        # In:  [batch_size, 7, 7, 64]
        # Out: [batch_size, 7*7*64]
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

        # In:  [batch_size, 7*7*64]
        # Out: [batch_size, 1024]
        dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

        # Dropout Layer
        dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=self.training)

        # The out of model which has not been softmax
        self.logits = tf.layers.dense(inputs=dropout, units=10)
        self.predictions = tf.argmax(self.logits, axis=1)

        # summary
        self.accuracy, self.update_op = tf.metrics.accuracy(self.labels, self.predictions)
        # self.summarizer.add_scalar("accuracy", self.accuracy)
        tf.summary.scalar("accuracy", self.accuracy)

        # Loss
        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=self.logits)
        # self.summarizer.add_scalar("loss", self.loss)
        tf.summary.scalar("loss", self.loss)

        # Train op
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        self.train_op = optimizer.minimize(
            loss=self.loss,
            global_step=self._global_step)

        # Merge Summary
        # self.summary = self.summarizer.merge()
        self.summary = tf.summary.merge_all()

    def train(self, dataset, *args, **kwargs):
        ds_iter = dataset.shuffle(1000).batch(self.config.n_batch).repeat(self.config.n_epoch).make_one_shot_iterator()
        features, labels = ds_iter.get_next()
        writer = tf.summary.FileWriter(
            r"D:\OneDrive\workspace\py\DL\tensorflow_template\examples\mnist\log\summary\train")

        while True:
            try:
                features_val, labels_val = self.sess.run([features, labels])
                loss, acc, summary, _, _ = self.sess.run(
                    [self.loss, self.accuracy, self.summary, self.update_op, self.train_op],
                    feed_dict={self.features: features_val,
                               self.labels: labels_val,
                               self.training: True})
                logger.info("Step {}: loss {}, accuracy {:.3}".format(self.global_step, loss, acc))
                self.summarizer.write_train(summary, self.global_step)
                writer.add_summary(summary, self.global_step)
            except tf.errors.OutOfRangeError:
                break

    def evaluate(self, dataset, *args, **kwargs):
        pass

    def predict(self, dataset, *args, **kwargs):
        pass


def get_config():
    config = Config('mnist')

    config.summary_dir = './log/summary'
    config.ckpt_dir = './log/checkpoint'

    config.n_batch = 16
    config.n_epoch = 5

    return config


def cnn():
    ds_train, ds_test = get_dataset()

    ds_iter = ds_train.shuffle(1000).batch(100).repeat(1).make_one_shot_iterator()

    f, l = ds_iter.get_next()
    logger.debug("dataset output type is {}".format(ds_train.output_types))

    ####

    features = tf.placeholder(tf.float32, [None, 28, 28, 1], name='features')
    labels = tf.placeholder(tf.int32, [None], name='labels')
    training = True

    global_step = tf.Variable(0, trainable=False, name='global_step')

    conv1 = tf.layers.conv2d(inputs=features,
                             filters=32,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu)

    # In:  [batch_size, 28, 28, 32]
    # Out: [batch_size, 14, 14, 32], here 14 = 28/strides = 28/2 = 14
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    # In:  [batch_size, 14, 14, 32]
    # Out: [batch_size, 14, 14, 64]
    conv2 = tf.layers.conv2d(inputs=pool1,
                             filters=64,
                             kernel_size=[5, 5],
                             padding="same",
                             activation=tf.nn.relu)

    # In:  [batch_size, 14, 14, 64]
    # Out: [batch_size, 7, 7, 64], where 7 is computed same as pool1
    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    # In:  [batch_size, 7, 7, 64]
    # Out: [batch_size, 7*7*64]
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    # In:  [batch_size, 7*7*64]
    # Out: [batch_size, 1024]
    dense = tf.layers.dense(inputs=pool2_flat, units=128, activation=tf.nn.relu)

    # Dropout Layer
    dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=training)

    # The out of model which has not been softmax
    logits = tf.layers.dense(inputs=dropout, units=10)
    predictions = tf.argmax(logits, axis=1)

    # summary
    accuracy, update_op = tf.metrics.accuracy(labels, predictions)
    tf.summary.scalar("accuracy", accuracy)

    # Loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    tf.summary.scalar("loss", loss)

    # Train op
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=global_step)

    # Merge Summary
    summary = tf.summary.merge_all()

    sess = tf.Session()
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    init_op.run(session=sess)

    writer = tf.summary.FileWriter(
        r"D:\OneDrive\workspace\py\DL\tensorflow_template\examples\mnist\log\summary\train")

    while True:
        try:
            ff, ll = sess.run([f, l])
            loss_val, acc, summary_val, _, _ = sess.run([loss, accuracy, summary, update_op, train_op],
                                                        feed_dict={features: ff, labels: ll})
            logger.info("Step {}: loss {}, accuracy {:.3}".format(global_step.eval(sess), loss_val, acc))
            writer.add_summary(summary_val, global_step.eval(sess))
        except tf.errors.OutOfRangeError:
            break


if __name__ == '__main__':
    """"""
    # train_images, train_labels, test_images, test_labels = get_data()

    ds_train, ds_test, ds_pred = get_dataset()

    config = get_config()

    model = MnistModel(config=config)

    model.train(ds_train)

    # cnn()
