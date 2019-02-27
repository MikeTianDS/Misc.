"""
The original code is from `https://github.com/dennybritz/cnn-text-classification-tf`
Here just re-organize the code in my template.
"""

import os
import time
import logging
import tensorflow as tf
from tensorflow_template import BaseModel, Config
from examples.cnn_text_classification import data_helper

logging.basicConfig(format='[%(name)s] : %(asctime)s : %(levelname)s : %(message)s',
                    level=logging.DEBUG)
logger = logging.getLogger(__name__)


class CnnText(BaseModel):
    """
    Use CNN for text classification. The original paper ref: http://arxiv.org/abs/1408.5882
    The model structure:
        - embedding layer
        - convolutional layer
        - max-pooling
        - softmax layer
    """

    def _init_graph(self):
        self.input_x = tf.placeholder(tf.int32, [None, self.config.sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, self.config.n_class], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        with tf.name_scope('embedding'):
            # Here use the random weights and adjust it when training.
            # If the original dataset is small or you want to apply the model to a wide application,
            # it should use some pre-trained word embedding, such as word2vec, GloVe or FastText.
            self.W = tf.Variable(
                tf.random_uniform([self.config.vocab_size, self.config.embedding_size], -1.0, 1.0),
                name="W")
            self.embedded_chars = tf.nn.embedding_lookup(self.W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)
            # shape (x, y) -> (x, y, 1)
            # The last dim expanded represent `channel` in the CNN input

        # Create cnn for different filter size:
        pooled_outputs = []
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.name_scope("cnn-%i" % filter_size):
                filter_shape = [filter_size, self.config.embedding_size, 1, self.config.n_filter]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.config.n_filter]), name="b")
                conv = tf.nn.conv2d(self.embedded_chars_expanded,
                                    W,
                                    strides=[1, 1, 1, 1],
                                    padding="VALID",
                                    name="conv")
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                pooled = tf.nn.max_pool(h,
                                        ksize=[1, self.config.sequence_length - filter_size + 1, 1, 1],
                                        strides=[1, 1, 1, 1],
                                        padding='VALID',
                                        name="pool")
                pooled_outputs.append(pooled)

        num_filters_total = self.config.n_filter * len(self.config.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (un-normalized) scores and predictions
        l2_loss = tf.constant(0.0)
        with tf.name_scope("output"):
            W = tf.get_variable("W",
                                shape=[num_filters_total, self.config.n_class],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.config.n_class]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + self.config.l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        # Train_op
        optimizer = tf.train.AdamOptimizer(self.config.learning_rate)
        grads_and_vars = optimizer.compute_gradients(self.loss)
        self.train_op = optimizer.apply_gradients(grads_and_vars, global_step=self._global_step)

        # Keep track of gradient values and sparsity
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # Output directory for models and summaries
        logger.info("Writing to {}\n".format(self.config.summary_dir))

        # Summaries for loss and accuracy
        loss_summary = tf.summary.scalar("loss", self.loss)
        acc_summary = tf.summary.scalar("accuracy", self.accuracy)

        # Train Summaries
        self.train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(self.config.summary_dir, "train")
        self.train_summary_writer = tf.summary.FileWriter(train_summary_dir, self.sess.graph)

        # Dev summaries
        self.dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
        dev_summary_dir = os.path.join(self.config.summary_dir, "dev")
        self.dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, self.sess.graph)

    def train(self, dataset, *args, **kwargs):
        """
        Args:
            dataset: (x_batch, y_batch)
        """
        x_batch, y_batch = zip(*dataset)
        feed_dict = {
            self.input_x: x_batch,
            self.input_y: y_batch,
            self.dropout_keep_prob: config.dropout_keep_prob
        }
        _, step, summaries, loss, accuracy = self.sess.run(
            [self.train_op, self._global_step, self.train_summary_op, self.loss, self.accuracy],
            feed_dict)
        logger.info("step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))
        self.train_summary_writer.add_summary(summaries, step)

    def evaluate(self, dataset, *args, **kwargs):
        """
        Args:
            dataset: (x_batch, y_batch)
        """
        x_batch, y_batch = dataset
        feed_dict = {
            self.input_x: x_batch,
            self.input_y: y_batch,
            self.dropout_keep_prob: 1.0
        }
        step, summaries, loss, accuracy = self.sess.run(
            [self._global_step, self.dev_summary_op, self.loss, self.accuracy],
            feed_dict)

        logger.info("Evaluate step {},\t loss {:g},\t acc {:g}".format(step, loss, accuracy))
        self.dev_summary_writer.add_summary(summaries, step)

    def predict(self, dataset, *args, **kwargs):
        pass


if __name__ == '__main__':
    """"""
    config = Config('cnn_text')
    config.sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)

    # Model Hyperparameters
    config.embedding_size = 300  # Dimensionality of character embedding
    # config.sequence_length = config.embedding_dim
    config.filter_sizes = [3, 4, 5]  # Comma-separated filter sizes
    config.n_filter = 128
    # config.n_class = 2
    config.vocab_size = 50000
    config.l2_reg_lambda = 0.0
    config.learning_rate = 1e-3

    # Train parameters
    config.n_batch = 256
    config.n_epoch = 50
    config.dropout_keep_prob = 0.5

    # Other
    # timestamp = str(int(time.time()))
    # config.log_dir = os.path.abspath(os.path.join(os.path.curdir, "log"))
    # if not os.path.exists(config.log_dir):
    #     os.makedirs(config.log_dir)
    # config.ckpt_dir = os.path.abspath(os.path.join(config.log_dir, "checkpoints"))
    # config.ckpt_prefix = os.path.join(config.ckpt_dir, "model")
    config.eval_step = 50  # Evaluate model on dev set after this many steps
    config.ckpt_step = 10  # Save model after this many steps

    # prepare the data
    x_train, x_test, y_train, y_test, vocab_processor = data_helper.get_dataset()
    vocab_processor.save(os.path.join(config.out_dir, "vocab"))  # binary file

    config.vocab_size = len(vocab_processor.vocabulary_)
    config.sequence_length = x_train.shape[1]  # 56,
    """Just the max length of the longest sentence in the dataset.
        If the length of some sentences is shorter than it, pad 0 at tail
    """
    config.n_class = y_train.shape[1]  # 2, pos and neg

    logger.debug("config.sequence_length = %i; config.n_class = %i" % (config.sequence_length, config.n_class))

    model = CnnText(config)
    model.load(r"D:\OneDrive\workspace\py\DL\tensorflow_template\examples\cnn_text_classification\log")
    logger.debug(model.global_step)

    batches = data_helper.batch_iter(
        list(zip(x_train, y_train)), config.n_batch, config.n_epoch)

    # train and eval
    for batch in batches:
        model.train(batch)
        if model.global_step % config.eval_step == 0:
            print()
            model.evaluate((x_test, y_test))
            print()
        if model.global_step % config.ckpt_step == 0:
            model.save()


