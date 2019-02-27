# Tensorflow Template

A deep learning template for tensorflow, of which the idea is from another project [MrGemy95/Tensorflow-Project-Template](https://github.com/MrGemy95/Tensorflow-Project-Template).

There is a new version which using the high-level tf-API - [imhuay/tensorflow_estimator](https://github.com/imhuay/tensorflow_estimator)

## Table Of Contents

<!-- TOC -->

- [Quick Start](#quick-start)
- [Examples](#examples)

<!-- /TOC -->

## Quick Start

The example is about the iris classification question. You can find the data and code at [examples/iris](./examples/iris)

### 1. Create a model class which inherit the [BaseModel](tensorflow_template/base/base_model.py)

All you need to do is finish the following four functions.

```python
class IrisModel(BaseModel):
    def _init_graph(self):
        pass

    def train(self, dataset, *args, **kwargs):
        pass

    def evaluate(self, dataset, *args, **kwargs):
        pass

    def predict(self, dataset, *args, **kwargs):
        pass
```

### 2. Build the graph
    
The basic part:
- the `tf.placeholder`
- the net
- the output (self.logits & self.prediction)
- the loss (self.loss)
- the train_op (self.train_op)

others:
- the metrics(such as `tf.metrics.accuracy`)
- the summary(ref `tf.summary.FileWriter`)
    
```python
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
```
    
### 3. The `train()`, `evaluate()` and `predict()`

The `dataset` is a [tf.data.Dataset](https://www.tensorflow.org/versions/master/get_started/datasets_quickstart) object.

Of course, the model does not limit to use it. You can choose the style of read data you like. 

```python
def train(self, dataset, buffer_size=1000, *args, **kwargs):
    for _ in range(self.config.n_epoch):
        # define the train epoch
        ds_iter = dataset.shuffle(buffer_size).batch(self.config.n_batch).make_one_shot_iterator()
        while True:
            # define the train step
            try:
                features, labels = self.sess.run(ds_iter.get_next())
                loss_val, _, _ = self.sess.run([self.loss, self.train_op, self.update_op],
                                               feed_dict={self.features: features, self.labels: labels})
                acc_val = self.sess.run(self.accuracy)
                logger.info("Step {}: loss {}, accuracy {:.3}".format(self.global_step, loss_val, acc_val))
            except tf.errors.OutOfRangeError:
                break
        self.save()

def evaluate(self, dataset, *args, **kwargs):
    self.mode = self.ModeKeys.EVAL
    ds_iter = dataset.shuffle(1000).batch(1).make_one_shot_iterator()

    acc_ret = dict()
    i = 1
    while True:
        try:
            features, labels = self.sess.run(ds_iter.get_next())
            prediction, _ = self.sess.run([self.prediction, self.update_op],
                                          feed_dict={self.features: features, self.labels: labels})
            logger.debug("labels is {}, prediction is {}".format(labels, prediction))
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
    ds_iter = dataset.shuffle(1000).batch(1).make_one_shot_iterator()

    pred_ret = []
    i = 1
    while True:
        try:
            features = self.sess.run(ds_iter.get_next())
            prediction = self.sess.run(self.prediction, feed_dict={self.features: features})
            pred_ret.append(prediction)
            logger.info("the prediction of No.{} is {}".format(i, prediction))
            i += 1
        except tf.errors.OutOfRangeError:
            break

    return np.array(pred_ret).flatten()
```
    
### 4. Run it
    
Here `Config` object is a subclass of Bunch object. If you want to use it, just `pip install bunch`.
    
```python
if __name__ == '__main__':
    logger.setLevel(logging.DEBUG)

    config = Config('ex', [4], 3)
    config.ckpt_dir = "./log/example_ckpt"
    if not os.path.exists(config.ckpt_dir):
        os.makedirs(config.ckpt_dir)
    config.n_batch = 64
    config.n_epoch = 100
    config.n_feature = [4]
    config.n_units = [10, 10]
    config.n_class = 3

    model = ExampleModel(config)

    from examples.iris.data_iris import *
    ds_train = get_dataset('train')
    ds_eval = get_dataset('eval')
    ds_predict = get_dataset('predict')

    logger.debug(model.global_step)

    model.load()
    logger.debug(model.global_step)

    model.train(ds_train)
    logger.debug(model.global_step)

    acc_ret = model.evaluate(ds_eval)
    print(acc_ret)

    pred_ret = model.predict(ds_predict)
    print(pred_ret)
```
    
## Examples

- iris classification [[code](./examples/iris)]
  - [TF Tutorials](https://www.tensorflow.org/get_started/get_started_for_beginners)
- mnist classification [[code](./examples/mnist)] (just train part)
  - [TF Tutorials](https://www.tensorflow.org/tutorials/layers)
- cnn-text-classification [[code](./examples/cnn_text_classification)]
  - The original [paper](http://arxiv.org/abs/1408.5882) and [github](https://github.com/yoonkim/CNN_sentence)
    
## Contributing

I always want to replace the `tf.placeholder` with `tf.data.Dataset` but no idea. 
You can see my trouble at [stackoverflow](https://stackoverflow.com/questions/49355553/how-to-write-a-template-for-most-tensorflow-deep-learning-project?answertab=votes#tab-top).
If you have a good resolvent, welcome it.

If you use the template to build some model. Welcome it to the examples.