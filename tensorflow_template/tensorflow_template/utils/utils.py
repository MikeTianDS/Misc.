import tensorflow as tf


def get_uninitialized_variables(sess):
    uninitialized_variables = list(tf.get_variable(name) for name in
                                   sess.run(tf.report_uninitialized_variables()))
    # session.run(tf.initialize_variables(uninitialized_variables))
    return uninitialized_variables

