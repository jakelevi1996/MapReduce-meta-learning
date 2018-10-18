import numpy as np
import tensorflow as tf


# TODO: add tensorboard summaries, especially for gradients


class ReptileRegressor():
    def __init__(
        self,
        # Network-params:
        input_dim=1, num_layers=2, num_hidden_units=15,
        hidden_activation=tf.tanh,
        # Meta-params:
        inner_learning_rate=0.01, outer_learning_rate=1,
        num_inner_steps=5, num_outer_steps=5,
    ):
        self.inner_learning_rate = inner_learning_rate
        # self.outer_learning_rate = outer_learning_rate|there is no outer rate
        self.num_inner_steps = num_inner_steps
        self.num_outer_steps = num_outer_steps

        # Define network
        # TODO: add regulariser
        # TODO: option for multiple layers
        self.x_placeholder = tf.placeholder(
            dtype=tf.float32,
            shape=(None, input_dim),
            name="Input"
        )
        self.hidden_op = tf.layers.dense(
            inputs=self.x_placeholder,
            units=num_hidden_units,
            activation=hidden_activation,
            name="Hidden_layer"
        )
        self.y_op = tf.layers.dense(
            inputs=self.hidden_op,
            units=1,
            name="Output"
        )

        # Define loss and optimiser
        self.y_placeholder = tf.placeholder(
            dtype=tf.float32,
            shape=(None, 1),
            name="Target"
        )
        self.loss_op = tf.losses.mean_squared_error(
            labels=self.y_placeholder,
            predictions=self.y_op
        )
        self.optimizer = tf.train.AdamOptimizer(inner_learning_rate)
        self.train_op = self.optimizer.minimize(self.loss_op)

        # Create summaries, for visualising in Tensorboard
        # ...
        
        # ***
        # NB the following statements all depend on calls to
        # tf.global_variables(), so any tf.Variables added to the graph after
        # them will not be taken into account (NB tf.Operations and
        # tf.placeholders are not instances of tf.Variables):
        # ***
        # oh look it's an init op
        self.init_op = tf.initializers.global_variables()
        # (does this want to be defined here, or within a method?):
        self.uninitialised_op = tf.report_uninitialized_variables()
        # ... <descriptive comment>
        self.global_vars = tf.global_variables()
        # ... etc
        self.parameter_placeholder_list = [
            tf.placeholder(dtype=tf.float32) for _ in self.global_vars
        ]
        # var.assign actually returns a Tensor, not an Operation...
        # see https://www.tensorflow.org/api_docs/python/tf/Variable#assign
        # ... but I like to think of assignment as an operation
        self.parameter_assign_op_list = [
            var.assign(placeholder) for (var, placeholder) in zip(
                self.global_vars, self.parameter_placeholder_list
            )
        ]
        # ...
        # NB axis must be set to zero because of the list-comprehension...
        self.reduced_mean_tensor_list = [
            tf.reduce_mean(
                input_tensor=placeholder,
                axis=0
            ) for placeholder in self.parameter_placeholder_list
        ]
        # Can insert an operation here in between reduce_mean and assign, EG
        # Bayesian inference for parameters by using previous value as a prior
        self.assign_reduced_mean_op_list = [
            var.assign(reduced_mean) for (var, reduced_mean) in zip(
                self.global_vars, self.reduced_mean_tensor_list
            )
        ]

    def initialise_variables(self, sess):
        sess.run(self.init_op)

    def predict(self, sess, x):
        predictions = sess.run(
            fetches=self.y_op,
            feed_dict={self.x_placeholder: x}
        )
        return predictions
    
    def eval_loss(self, sess, x, y):
        sess.run(
            fetches=self.loss_op,
            feed_dict={
                self.x_placeholder: x,
                self.y_placeholder: y
            }
        )

    def training_step(self, sess, x, y):
        sess.run(
            fetches=self.train_op,
            feed_dict={
                self.x_placeholder: x,
                self.y_placeholder: y
            }
        )
    
    def get_global_vars(self, sess):
        return sess.run(self.global_vars)
    
    def set_global_vars(self, sess, global_vars):
        # `global_vars` must come from a call to the `get_global_vars` method;
        # could maybe use an `assert` statement for the shape of each param?
        sess.run(
            fetches=self.parameter_assign_op_list,
            feed_dict={
                placeholder: value for placeholder, value in zip(
                    self.parameter_placeholder_list, global_vars
                )
            }
        )
    
    def adapt(self, sess, x, y):
        # In reality this will be some steps of gradient descent
        for _ in range(self.num_inner_steps):
            self.training_step(sess, x, y)
    
    def meta_update(self, sess, params_list):
        sess.run(
            fetches=self.assign_reduced_mean_op_list,
            feed_dict={
                placeholder: [
                    params[val_index] for params in params_list
                ] for (placeholder, val_index) in zip(
                    self.parameter_placeholder_list,
                    range(len(self.parameter_placeholder_list))
                )
            }
        )

class Maml():
    pass
