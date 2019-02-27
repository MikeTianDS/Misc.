"""
A summary writer helper
"""
import os
import tensorflow as tf
from huaytools import Bunch


class Summary(object):
    summary_op = Bunch(scalar=tf.summary.scalar,
                       histogram=tf.summary.histogram,
                       image=tf.summary.image,
                       # audio=tf.summary.audio,
                       tensor_summary=tf.summary.tensor_summary)

    def __init__(self, summary_dir, sess):
        """"""
        self.summary_dir = summary_dir
        self.summary_list = []
        self.sess = sess

        train_summary_dir = os.path.join(self.summary_dir, 'train')
        os.makedirs(train_summary_dir, exist_ok=True)
        self.train_writer = tf.summary.FileWriter(train_summary_dir)

        eval_summary_dir = os.path.join(self.summary_dir, 'eval')
        os.makedirs(eval_summary_dir, exist_ok=True)
        self.eval_writer = tf.summary.FileWriter(eval_summary_dir)

    def merge(self, summary_list=None):
        """
        Get summary_merge_op from `self.summary_list` default.

        Args:
            summary_list:

        Returns:
            merge_op

        """
        if summary_list is None:
            summary_list = self.summary_list

        merge_op = tf.summary.merge(summary_list)
        return merge_op

    def write(self, summary, global_step=None, summary_type='train'):
        """
        Write the summary values to file for TensorBoard

        Args:
            summary: the return value of `sess.run(merge_op)`

        Returns:

        """
        if summary_type == 'train':
            # train_summary_dir = os.path.join(self.summary_dir, 'train')
            # os.makedirs(train_summary_dir, exist_ok=True)
            # train_writer = tf.summary.FileWriter(train_summary_dir)
            self.train_writer.add_summary(summary, global_step)
        else:
            # eval_summary_dir = os.path.join(self.summary_dir, 'eval')
            # os.makedirs(eval_summary_dir, exist_ok=True)
            # eval_writer = tf.summary.FileWriter(eval_summary_dir)
            self.eval_writer.add_summary(summary, global_step)

    def write_train(self, summary, global_step=None):
        self.write(summary, global_step, summary_type='train')

    def write_eval(self, summary, global_step=None):
        self.write(summary, global_step, summary_type='eval')

    def add(self, name, tensor, op_type="scalar"):
        tmp_summary = self.summary_op[op_type](name, tensor)
        self.summary_list.append(tmp_summary)
        return tmp_summary

    def add_scalar(self, name, tensor, collections=None, family=None):
        scalar_summary = tf.summary.scalar(name, tensor,
                                           collections=collections, family=family)
        self.summary_list.append(scalar_summary)
        return scalar_summary

    def add_histogram(self, name, tensor, collections=None, family=None):
        histogram_summary = tf.summary.histogram(name, tensor,
                                                 collections=collections, family=family)
        self.summary_list.append(histogram_summary)
        return histogram_summary

    def add_image(self, name, tensor, max_outputs=3, collections=None, family=None):
        image_summary = tf.summary.image(name, tensor,
                                         max_outputs=max_outputs, collections=collections, family=family)
        self.summary_list.append(image_summary)
        return image_summary

    def add_audio(self, name, tensor, sample_rate, max_outputs=3, collections=None, family=None):
        audio_summary = tf.summary.audio(name, tensor, sample_rate,
                                         max_outputs=max_outputs, collections=collections, family=family)
        self.summary_list.append(audio_summary)
        return audio_summary

    def add_tensor_summary(self, name, tensor,
                           summary_description=None, collections=None, summary_metadata=None, family=None,
                           display_name=None):
        tensor_summary = tf.summary.tensor_summary(name, tensor,
                                                   summary_description=summary_description,
                                                   collections=collections,
                                                   summary_metadata=summary_metadata,
                                                   family=family,
                                                   display_name=display_name)
        self.summary_list.append(tensor_summary)
        return tensor_summary
