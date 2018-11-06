import logging
import tensorflow as tf
import numpy as np
from multiprocessing.pool import Pool
from time import time

from models.neuralmodels import ReptileRegressor
from data import tasksets


def inner_update(initial_params, task):
    # Create an instance of the model:
    model = ReptileRegressor()
    # Start a session:
    with tf.Session() as sess:
        # Initialise parameters to meta-parameters:
        model.set_global_vars(sess, initial_params)
        # Check that all variables have been initialised:
        model.report_uninitialised(sess)
        # Perform fast adaptation:
        model.adapt(sess, task.train)
        # Retrieve the learned task-specific parameters:
        task_specific_params = model.get_global_vars(sess)
        # Evaluate test loss:
        test_loss = model.eval_loss(sess, task.test.x, task.test.y)
    
    # Have to reset the default graph, otherwise variables in this worker's
    # graph persist even after the session ends and those variables lose their
    # values and the worker starts handling a new task:
    tf.reset_default_graph()
    
    # Return the task specific parameters to the outer loop
    return task_specific_params, test_loss

def outer_update(model, sess, task_set, pool):
    # Retrieve meta-parameters, ready to pass to the inner update:
    meta_params = model.get_global_vars(sess)
    # Create list of argument tuples, one for each process:
    num_tasks = task_set.X_train.shape[0]
    inner_arg_list = [
        (meta_params, task_set.get_task(i)) for i in range(num_tasks)
    ]
    # Map arguments onto a list of task-specific parameters and test_losses:
    results = pool.starmap(inner_update, inner_arg_list)
    # Separate results:
    task_specific_params_list, loss_list = zip(*results)
    # Log training task specific test loss:
    logging.info("Loss list: " + str(loss_list))
    logging.info("Mean = " + str(np.mean(loss_list)))
    # Reduce list of task-specific parameters into meta-parameters:
    model.meta_update(sess, task_specific_params_list)


def train(model, task_set, num_outer_steps=10):
    # Initialise process pool and TensorFlow session:
    with Pool() as pool, tf.Session() as sess:
        logging.info("Warming up the pool...")
        pool.map(int, [0])
        start_time = time()
        logging.info("Initialising variables...")
        model.initialise_variables(sess)
        model.report_uninitialised(sess)

        for epoch in range(num_outer_steps):
            logging.info("Epoch = {}".format(epoch))
            outer_update(model, sess, task_set, pool)
        
        # Retreive meta-params and save model:
        # meta_params = model.get_global_vars(sess)
        logging.info("Time taken = {:.2f}".format(time() - start_time))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    logging.info("Loading data...")
    filename = tasksets.DEFAULT_FILE_NAME
    meta_set = tasksets.SinusoidalMetaSet(filename)
    
    logging.info("Creating model...")
    model = ReptileRegressor()

    train(model, meta_set.train_tasks, num_outer_steps=25)
