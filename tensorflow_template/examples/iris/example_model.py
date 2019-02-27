import os
import logging
import tensorflow as tf

from tensorflow_template import Config
from tensorflow_template import BaseModel

logger = logging.getLogger(__name__)
logging.basicConfig(format='[%(name)s] : %(asctime)s : %(levelname)s : %(message)s',
                    level=logging.INFO)


class ExampleModel(BaseModel):
    """"""

    def _init_graph(self):
        # 1. define the `tf.placeholder`
        self.features = tf.placeholder(tf.float32, [None] + self.config.n_feature, 'features')
        self.labels = tf.placeholder(tf.int32, [None], 'labels')

        # 2. define the net
        net = self.features  # input_layer
        for units in self.config.n_units:
            net = tf.layers.dense(net, units=units, activation=tf.nn.relu)
            # net = tf.layers.Dense(units=units, activation=tf.nn.relu)(net)

        # the output
        self.logits = tf.layers.dense(net, self.config.n_class, activation=None)
        self.prediction = tf.argmax(self.logits, axis=1)

        self.accuracy, self.update_op = tf.metrics.accuracy(labels=self.labels,
                                                            predictions=self.prediction,
                                                            name='acc_op')

        # 3. define the loss
        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.labels, logits=self.logits)

        # 4. define the train_op
        self.optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
        self.train_op = self.optimizer.minimize(self.loss, global_step=self._global_step)

    def train(self, dataset, buffer_size=1000, *args, **kwargs):
        ds_iter = dataset.shuffle(buffer_size).batch(self.config.n_batch).make_one_shot_iterator()
        features, labels = ds_iter.get_next()
        for _ in range(self.config.n_epoch):
            # define the train epoch
            while True:
                # define the train step
                try:
                    # it's quite a error coding
                    # features, labels = self.sess.run(ds_iter.get_next())  
                    features_val, labels_val = self.sess.run([features, labels])
                    loss_val, _, _ = self.sess.run([self.loss, self.train_op, self.update_op],
                                                   feed_dict={self.features: features_val, self.labels: labels_val})
                    acc_val = self.sess.run(self.accuracy)
                    logger.info("Step {}: loss {}, accuracy {:.3}".format(self.global_step, loss_val, acc_val))
                except tf.errors.OutOfRangeError:
                    break
            self.save()

    def evaluate(self, dataset, *args, **kwargs):
        self.mode = self.ModeKeys.EVAL
        ds_iter = dataset.batch(1).make_one_shot_iterator()
        features, labels = ds_iter.get_next()

        acc_ret = dict()
        i = 1
        while True:
            try:
                features_val, labels_val = self.sess.run([features, labels])
                prediction, _ = self.sess.run([self.prediction, self.update_op],
                                              feed_dict={self.features: features_val, self.labels: labels_val})
                logger.debug("labels is {}, prediction is {}".format(labels_val, prediction))
                # run `update_op` first, then run the `accuracy`
                acc_val = self.sess.run(self.accuracy)
                logger.info('Accuracy is {:.3} of {} test samples'.format(acc_val, i))
                acc_ret[i] = acc_val
                i += 1
            except tf.errors.OutOfRangeError:
                break

        return acc_ret

    def predict(self, dataset, *args, **kwargs):
        self.mode = self.ModeKeys.PREDICT
        ds_iter = dataset.batch(1).make_one_shot_iterator()
        features = ds_iter.get_next()

        pred_ret = []
        i = 1
        while True:
            try:
                features_val = self.sess.run(features)
                prediction = self.sess.run(self.prediction, feed_dict={self.features: features_val})
                pred_ret.append(prediction)
                logger.info("the prediction of No.{} is {}".format(i, prediction))
                i += 1
            except tf.errors.OutOfRangeError:
                break

        return np.array(pred_ret).flatten()


if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)

    config = Config('ex')
    if not os.path.exists(config.ckpt_dir):
        os.makedirs(config.ckpt_dir)
    config.n_batch = 64
    config.n_epoch = 5
    config.n_feature = [4]
    config.n_units = [10, 10]
    config.n_class = 3

    model = ExampleModel(config)

    from examples.iris.data_iris import *

    ds_train = get_dataset('train')
    logger.debug(ds_train.output_shapes)
    ds_eval = get_dataset('eval')
    logger.debug(ds_eval.output_shapes)
    ds_predict = get_dataset('predict')
    logger.debug(ds_predict.output_shapes)

    logger.debug(model.global_step)

    model.load()
    logger.debug(model.global_step)

    model.train(ds_train)
    logger.debug(model.global_step)

    acc_ret = model.evaluate(ds_eval)
    print(acc_ret)

    pred_ret = model.predict(ds_predict)
    print(pred_ret)

    # from tensorflow.python.tools import inspect_checkpoint as chkp
    #
    # latest_ckpt = tf.train.latest_checkpoint(config.ckpt_dir)
    #
    # chkp.print_tensors_in_checkpoint_file(latest_ckpt,
    #                                       tensor_name='', all_tensors=True, all_tensor_names=True)
