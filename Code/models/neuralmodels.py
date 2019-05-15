import numpy as np
import tensorflow as tf


# TODO: add tensorboard summaries, especially for gradients


class NeuralRegressor():
    def __init__(
        self, input_dim=1, num_hidden_units=[20, 20, 20],
        hidden_activation=tf.tanh, learning_rate=0.01
    ):
        self.initialise_network(
            input_dim, num_hidden_units, hidden_activation, learning_rate
        )
        self.initialise_tensorboard_summaries()
        self.initialise_global_ops()
    
    def initialise_network(
        self, input_dim, num_hidden_units, hidden_activation, learning_rate, 
    ):
        # TODO: add regulariser
        # TODO: allow multiple layers by specifying num_hidden_units as a list

        # Define network
        self.x_placeholder = tf.placeholder(
            dtype=tf.float32,
            shape=(None, input_dim),
            name="Input"
        )
        # If number of hidden units is an integer, create a single-layer net:
        if type(num_hidden_units) is int:
            self.hidden_ops = [tf.layers.dense(
                inputs=self.x_placeholder,
                units=num_hidden_units,
                activation=hidden_activation,
                name="Hidden_layer"
            )]
        # If number of hidden units is a list, create multiple hidden layers:
        else:
            try:
                self.hidden_ops = [tf.layers.dense(
                    inputs=self.x_placeholder,
                    units=num_hidden_units[0],
                    activation=hidden_activation,
                    name="Hidden_layer_0"
                )]
                for n, h in enumerate(num_hidden_units[1:]):
                    self.hidden_ops.append(tf.layers.dense(
                        inputs=self.hidden_ops[n],
                        units=h,
                        activation=hidden_activation,
                        name="Hidden_layer_{}".format(n+1)
                    ))
            except:
                # TODO: use better error reporting + message
                raise ValueError("Bad number of hidden units specified.")
        # Output layer:
        self.y_op = tf.layers.dense(
            inputs=self.hidden_ops[-1],
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
    
    def initialise_tensorboard_summaries(self):
        # TODO: add tensorboard summaries, and methods for logging them
        pass

    def initialise_global_ops(self,):
        """
        NB the following statements all depend on calls to
        tf.global_variables(), so any tf.Variables added to the graph after
        running this method not be taken into account (NB tf.Operations and
        tf.placeholders are not instances of tf.Variables):

        TODO: update this with better docstring + comments
        """
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
        self.parameter_assign_op_list = [
            var.assign(placeholder) for (var, placeholder) in zip(
                self.global_vars, self.parameter_placeholder_list
            )
        ]
    
    # TODO: add save/restore
    def save_params(self, filename): pass
    def restore_params(self, filename): pass

    def initialise_variables(self, sess): sess.run(self.init_op)

    def predict(self, sess, x):
        predictions = sess.run(
            fetches=self.y_op,
            feed_dict={self.x_placeholder: x}
        )
        return predictions
    
    def eval_loss(self, sess, x, y):
        loss = sess.run(
            fetches=self.loss_op,
            feed_dict={self.x_placeholder: x, self.y_placeholder: y}
        )
        return loss

    def training_step(self, sess, x, y):
        sess.run(
            fetches=self.train_op,
            feed_dict={self.x_placeholder: x, self.y_placeholder: y}
        )
    
    def get_global_vars(self, sess): return sess.run(self.global_vars)
    
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
    
    def report_uninitialised(self, sess):
        pass
        # uninitialised_list = sess.run(self.uninitialised_op)
        # if len(uninitialised_list) > 0:
        #     print("Remaining uninitialised variables:")
        #     for var in uninitialised_list:
        #         print("  - {}".format(var))

class ReptileRegressor(NeuralRegressor):
    def __init__(
        self,
        # Network-params:
        input_dim=1, num_hidden_units=15, hidden_activation=tf.tanh,
        # Meta-params:
        inner_learning_rate=0.01, num_inner_steps=32,
        # outer_learning_rate=1, num_outer_steps=5,
    ):
        # self.inner_learning_rate = inner_learning_rate
        # self.outer_learning_rate = outer_learning_rate|there is no outer rate
        # self.num_outer_steps = num_outer_steps
        self.num_inner_steps = num_inner_steps

        self.initialise_network(
            input_dim, num_hidden_units, hidden_activation, inner_learning_rate
        )
        self.initialise_tensorboard_summaries()
        self.initialise_global_ops()
        self.initialise_global_reduce_ops()

    def initialise_global_reduce_ops(self,):
        """
        NB these meta-learning specific method depends on attributes defined in
        `self.initialise_global_ops()`, so that method must be run first.
        
        The Operations defined in this method refer to reducing operations
        applied during the outer update to the results of the inner update

        TODO: update this with better docstring + comments
        """
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
    
    def adapt(self, sess, data_set):
        for _ in range(self.num_inner_steps):
            self.training_step(sess, data_set.x, data_set.y)
    
    def meta_update(self, sess, params_list):
        sess.run(
            fetches=self.assign_reduced_mean_op_list,
            feed_dict={
                # Each placeholder corresponds to a single val_index in
                # `self.parameter_placeholder_list`, and multiple parameters
                # from each fast adaptation that are reduced into a single
                # parameter:
                placeholder: [
                    params[val_index] for params in params_list
                ] for (val_index, placeholder) in enumerate(
                    self.parameter_placeholder_list
                )
            }
        )

class Maml():
    pass

class HRM():
    pass

class BinarySetRegressor():
    def __init__(
        self, num_hidden_units=250, hidden_activation=tf.nn.relu,
        num_hidden_layers_pre_reduce=2, num_hidden_layers_post_reduce=4,
        learning_rate=1e-3
    ):
        # Input placeholders
        self.dark_inputs = tf.placeholder(tf.float32, shape=[None, None, 2])
        self.light_inputs = tf.placeholder(tf.float32, shape=[None, None, 2])
        self.test_pixels_x = tf.placeholder(tf.float32, shape=[None, None, 2])
        self.test_pixels_y = tf.placeholder(tf.float32, shape=[None, None, 1])
        num_test_pixels = tf.shape(self.test_pixels_x)[1]
        # Network for dark pixels
        darknet_out = self.pre_reduce_layers(
            self.dark_inputs, hidden_activation, num_hidden_layers_pre_reduce,
            num_hidden_units, num_test_pixels, name_prefix="darknet"
        )
        # Network for light pixels
        lightnet_out = self.pre_reduce_layers(
            self.light_inputs, hidden_activation, num_hidden_layers_pre_reduce,
            num_hidden_units, num_test_pixels, name_prefix="lightnet"
        )
        # Concatenate vectors
        concat_vector = tf.concat(
            [darknet_out, lightnet_out, self.test_pixels_x], axis=2,
            name="concat_vector"
        )
        # Dense layers for output from reduce operations to final output
        post_reduce_hidden_layers = [tf.layers.dense(
            inputs=concat_vector, units=num_hidden_units,
            activation=hidden_activation, name="post_reduce_1"
        )]
        for i in range(1, num_hidden_layers_post_reduce - 1):
            post_reduce_hidden_layers.append(tf.layers.dense(
                inputs=post_reduce_hidden_layers[-1], units=num_hidden_units,
                activation=hidden_activation, name="post_reduce_"+str(i+1)
            ))
        logits = tf.layers.dense(
            inputs=post_reduce_hidden_layers[-1], units=1, name="logits"
        )
        self.output = tf.sigmoid(logits, name="output")


        self.loss_op = tf.losses.sigmoid_cross_entropy(
            self.test_pixels_y, logits
        )
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(
            self.loss_op
        )

        self.init_op = tf.global_variables_initializer()

        # Create summaries, for visualising in Tensorboard
        self.summary_op = tf.summary.scalar("Loss", self.loss_op)

    def pre_reduce_layers(
        self, inputs, hidden_activation, num_hidden_layers, num_hidden_units,
        num_test_pixels, name_prefix
    ):
        hidden_layers = [tf.layers.dense(
            inputs=inputs, units=num_hidden_units,
            activation=hidden_activation, name=name_prefix+"_hidden_1"
        )]
        for i in range(1, num_hidden_layers):
            hidden_layers.append(tf.layers.dense(
                inputs=hidden_layers[-1], units=num_hidden_units,
                activation=hidden_activation,
                name=name_prefix+"_hidden_"+str(i+1)
            ))
        reduce_op = tf.reduce_mean(
            hidden_layers[-1], axis=1, keepdims=True,
            name=name_prefix+"_reduced"
        )
        output = tf.tile(
            reduce_op, [1, num_test_pixels, 1], name=name_prefix+"_tiled"
        )
        return output
    
    def initialise_variables(self, sess): sess.run(self.init_op)
    
    def training_step(self, sess, x_dark_train, x_light_train, x_test, y_test):
        loss_val, summary_val, _ = sess.run(
            (self.loss_op, self.summary_op, self.train_op), feed_dict={
                self.dark_inputs: x_dark_train,
                self.light_inputs: x_light_train,
                self.test_pixels_x: x_test,
                self.test_pixels_y: y_test,
            }
        )
        return loss_val, summary_val
    
    def eval(
        self, sess, x_dark_train, x_light_train, x_test, output_shape=-1,
        num_batches=1
    ):
        return sess.run(self.output, feed_dict={
            self.dark_inputs: x_dark_train.reshape(num_batches, -1, 2),
            self.light_inputs: x_light_train.reshape(num_batches, -1, 2),
            self.test_pixels_x: x_test.reshape(num_batches, -1, 2),
        }).reshape(output_shape)

class ContinuousSetRegressor():
    def __init__(
        self, num_hidden_units=250, hidden_activation=tf.nn.relu,
        num_hidden_layers_pre_reduce=2, num_hidden_layers_post_reduce=4,
        learning_rate=1e-3, loss_func=tf.losses.mean_squared_error
    ):
        self.create_hidden_layers(
            num_hidden_units, hidden_activation,
            num_hidden_layers_pre_reduce, num_hidden_layers_post_reduce
        )
        # Define output
        logits = tf.layers.dense(
            inputs=self.post_reduce_hidden_layers[-1], units=1, name="logits"
        )
        # Maybe should use linear output and MSE?
        # self.output = tf.sigmoid(logits, name="output")
        self.output = logits

        # Define loss, training and initialisation operations
        self.loss_op = loss_func(self.y_eval, logits )
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(
            self.loss_op
        )

        self.init_op = tf.global_variables_initializer()

        # Create summaries, for visualising in Tensorboard
        self.summary_op = tf.summary.scalar("Loss", self.loss_op)

        # Create saver object
        self.saver = tf.train.Saver()
    
    def create_hidden_layers(
        self, num_hidden_units, hidden_activation,
        num_hidden_layers_pre_reduce, num_hidden_layers_post_reduce
    ):
        # Input placeholders
        self.x_condition = tf.placeholder(tf.float32, shape=[None, None, 2])
        self.y_condition = tf.placeholder(tf.float32, shape=[None, None, 1])
        self.x_eval = tf.placeholder(tf.float32, shape=[None, None, 2])
        self.y_eval = tf.placeholder(tf.float32, shape=[None, None, 1])
        num_eval_points = tf.shape(self.x_eval)[1]
        # Concatenate conditioning points
        conditioning_set = tf.concat(
            [self.x_condition, self.y_condition], axis=2, name="condition_set"
        )
        # Dense layers before reduction
        pre_reduce_hidden_layers = [tf.layers.dense(
            inputs=conditioning_set, units=num_hidden_units,
            activation=hidden_activation, name="pre_reduce_1"
        )]
        for i in range(1, num_hidden_layers_pre_reduce):
            pre_reduce_hidden_layers.append(tf.layers.dense(
                inputs=pre_reduce_hidden_layers[-1], units=num_hidden_units,
                activation=hidden_activation, name="pre_reduce_"+str(i+1)
            ))
        # Reduce, tile and concatenate
        reduce_op = tf.reduce_mean(
            pre_reduce_hidden_layers[-1], axis=1, keepdims=True, name="reduce"
        )
        tile_op = tf.tile(
            reduce_op, [1, num_eval_points, 1], name="tile"
        )
        concat_op = tf.concat(
            [tile_op, self.x_eval], axis=2, name="concatenate"
        )
        # Dense layers for output from reduce operations to final output
        self.post_reduce_hidden_layers = [tf.layers.dense(
            inputs=concat_op, units=num_hidden_units,
            activation=hidden_activation, name="post_reduce_1"
        )]
        for i in range(1, num_hidden_layers_post_reduce - 1):
            self.post_reduce_hidden_layers.append(tf.layers.dense(
                self.post_reduce_hidden_layers[-1], num_hidden_units,
                activation=hidden_activation, name="post_reduce_"+str(i+1)
            ))
    
    def initialise_variables(self, sess): sess.run(self.init_op)
    
    def training_step(self, sess, x_condition, y_condition, x_eval, y_eval):
        sess.run(
            self.train_op, feed_dict={
                self.x_condition: x_condition,
                self.y_condition: y_condition,
                self.x_eval: x_eval,
                self.y_eval: y_eval,
            }
        )
    
    def eval_output(
        self, sess, x_condition, y_condition, x_eval, num_batches=1
    ):
        return sess.run(self.output, feed_dict={
            self.x_condition: x_condition.reshape(num_batches, -1, 2),
            self.y_condition: y_condition.reshape(num_batches, -1, 1),
            self.x_eval: x_eval.reshape(num_batches, -1, 2),
        })
    
    def eval_loss(self, sess, x_condition, y_condition, x_eval, y_eval):
        loss_val, summary_val = sess.run(
            [self.loss_op, self.summary_op], feed_dict={
                self.x_condition: x_condition,
                self.y_condition: y_condition,
                self.x_eval: x_eval,
                self.y_eval: y_eval,
            }
        )
        return loss_val, summary_val
    
    def save(self, sess, save_path):
        save_path = self.saver.save(sess, save_path)
        print("Model saved as < {} >".format(save_path))
        return save_path
    
    def restore(self, sess, save_path): self.saver.restore(sess, save_path)


class ProbabilisticCSR(ContinuousSetRegressor):
    def __init__(
        self, num_hidden_units=250, hidden_activation=tf.nn.relu,
        num_hidden_layers_pre_reduce=2, num_hidden_layers_post_reduce=4,
        num_hidden_layers_pre_output=2, learning_rate=1e-6,
        loss_function="density", var_clip=0
    ):
        self.create_hidden_layers(
            num_hidden_units, hidden_activation,
            num_hidden_layers_pre_reduce, num_hidden_layers_post_reduce
        )
        # Define mean, log-variance, and STD
        pre_mean_hidden_layers = [tf.layers.dense(
            inputs=self.post_reduce_hidden_layers[-1], units=num_hidden_units,
            activation=hidden_activation, name="pre_mean_1"
        )]
        for i in range(1, num_hidden_layers_pre_output):
            pre_mean_hidden_layers.append(tf.layers.dense(
                inputs=pre_mean_hidden_layers[-1], units=num_hidden_units,
                activation=hidden_activation, name="pre_mean_"+str(i+1)
            ))
        self.mean = tf.layers.dense(
            inputs=pre_mean_hidden_layers[-1], units=1, name="mean"
        )

        pre_logvar_hidden_layers = [tf.layers.dense(
            inputs=self.post_reduce_hidden_layers[-1], units=num_hidden_units,
            activation=hidden_activation, name="pre_logvar_1"
        )]
        for i in range(1, num_hidden_layers_pre_output):
            pre_logvar_hidden_layers.append(tf.layers.dense(
                inputs=pre_logvar_hidden_layers[-1], units=num_hidden_units,
                activation=hidden_activation, name="pre_logvar_"+str(i+1)
            ))
        self.log_var = tf.layers.dense(
            inputs=pre_logvar_hidden_layers[-1], units=1, name="log_var"
        )
        
        self.std = tf.exp(0.5 * self.log_var)

        # Define loss, training and initialisation operations
        squared_error = tf.square(self.y_eval - self.mean, name="square-error")
        
        # Initialise user-specified loss function
        if loss_function == "density":
            # scaled log-gaussian density
            self.loss_op = tf.reduce_mean(
                self.log_var + squared_error * tf.exp(-1*self.log_var),
                name="density_loss"
            )
        elif loss_function == "clipped density":
            # Use soft-clip to regularise gradients
            def soft_clip(x, y=var_clip): return tf.log(tf.exp(x) + np.exp(y))
            self.loss_op = tf.reduce_mean(
                soft_clip(self.log_var)+squared_error*tf.exp(-1*self.log_var),
                name="density_loss"
            )
        elif loss_function == "MSE":
            # Ignore variance predictions, just use MSE
            self.loss_op = tf.reduce_mean(squared_error, name="mse_loss")
        else: raise ValueError("Incorrect loss function specified for PCSR")
        
        self.train_op = tf.train.AdamOptimizer(learning_rate).minimize(
            self.loss_op
        )

        self.init_op = tf.global_variables_initializer()

        # Create summaries, for visualising in Tensorboard
        tf.summary.scalar("Loss", self.loss_op)
        tf.summary.histogram("Mean", self.mean)
        tf.summary.histogram("Log-variance", self.log_var)
        self.summary_op = tf.summary.merge_all()

        # Create saver object
        self.saver = tf.train.Saver()

    def eval_output(
        self, sess, x_condition, y_condition, x_eval, num_batches=1
    ):
        return sess.run(self.mean, feed_dict={
            self.x_condition: x_condition.reshape(num_batches, -1, 2),
            self.y_condition: y_condition.reshape(num_batches, -1, 1),
            self.x_eval: x_eval.reshape(num_batches, -1, 2),
        })
    
    def eval_mean_std(
        self, sess, x_condition, y_condition, x_eval, num_batches=1
    ):
        mean, std = sess.run([self.mean, self.std], feed_dict={
            self.x_condition: x_condition.reshape(num_batches, -1, 2),
            self.y_condition: y_condition.reshape(num_batches, -1, 1),
            self.x_eval: x_eval.reshape(num_batches, -1, 2),
        })
        return mean, std

    def sample(
        self, sess, x_condition, y_condition, x_eval, num_batches=1
    ):
        mean, std = self.eval_mean_std(
            sess, x_condition, y_condition, x_eval, num_batches=1
        )
        # print(mean, std, x_eval.shape)
        return np.random.normal(mean, std)