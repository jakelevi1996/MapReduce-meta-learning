import numpy as np
import tensorflow as tf

# TODO: 2 classes: Regressor, and TrainableRegressor which inherits from
# Regressor? Because the meta-parameters won't be using any of tensorflow's
# built-in training procedures, they'll all be manual based on meta-learning
# algorithms

# TODO: add tensorboard summaries, especially for gradients

class Regressor():
    
    def __init__(
        self,
        input_dim=1, num_hidden_units=15,
        hidden_activation=tf.tanh,
        learning_rate=0.01,
        ):
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
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_op = self.optimizer.minimize(self.loss_op)

        # Create op for initialising variables
        self.init_op = tf.global_variables_initializer()

        # Create summaries, for visualising in Tensorboard
        # Training summaries:
        # tf.summary.scalar("Accuracy", accuracy)
        # variables = tf.trainable_variables()
        # gradients = tf.gradients(self.loss_op, variables)
        # self.train_summary_op = tf.summary.merge([
        #     tf.summary.scalar("Train_loss", self.loss_op),
        #     *[tf.summary.histogram(parse_name(v.name), v) for v in variables],
        #     *[tf.summary.histogram(parse_name(g.name), g) for g in gradients]
        # ])
        # # Test summaries:
        # self.test_summary_op = tf.summary.merge([
        #     tf.summary.scalar("Test_loss", self.loss_op),
        #     # <Other test summaries>
        # ])
        

    def initialize_variables(self, sess):
        sess.run(self.init_op)

    def predict(self, sess, x):
        predictions = sess.run(
            self.y_op,
            feed_dict={self.x_placeholder: x}
        )
        return predictions

    def training_step(self, sess, x, y):
        sess.run(
            self.train_op,
            feed_dict={
                self.x_placeholder: x,
                self.y_placeholder: y
            }
        )
    
    # def training_step_with_summary(self, sess, x_train, y_train):
    #     train_loss_val, train_summary_val, _ = sess.run([
    #         self.loss_op,
    #         self.train_summary_op,
    #         self.train_op],
    #         feed_dict={
    #             self.input_placeholder: x_train,
    #             self.labels_placeholder: y_train
    #         }
    #     )
    #     return train_loss_val, train_summary_val
    
    # def test_set_summary(self, sess, x_test, y_test):
    #     test_loss_val, test_summary_val = sess.run([
    #         self.loss_op,
    #         self.test_summary_op],
    #         feed_dict={
    #             self.input_placeholder: x_test,
    #             self.labels_placeholder: y_test
    #         }
    #     )
    #     return test_loss_val, test_summary_val

class ReptileRegressor():
    def __init__(
        self,
        input_dim=1,
        inner_learning_rate=0.001, outer_learning_rate=1,
        num_layers=2, num_units=15, hidden_activation=tf.tanh,
    ):
        self.inner_learning_rate = inner_learning_rate
        self.outer_learning_rate = outer_learning_rate

        # TODO: Make sure vairable_scope is correct and should be used instead
        # of name_scope
        with tf.variable_scope("Meta_parameters"):
            self.meta_network = Regressor()
        with tf.variable_scope("Task_specific_parameters"):
            self.task_specific_network = Regressor()
    
    def inner_update(
        self,

    ):
        pass
    
    def outer_update(
        self,
        
    ):
        pass
