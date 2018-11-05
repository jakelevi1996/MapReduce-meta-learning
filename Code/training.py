import logging
import tensorflow as tf
from multiprocessing.pool import Pool
from time import time

from models.neuralmodels import ReptileRegressor
from data import tasksets


def inner_update(initial_params, data_set):
    # Create an instance of the model:
    model = ReptileRegressor()
    # Start a session:
    with tf.Session() as sess:
        # Set initial parameters:
        model.set_global_vars(sess, initial_params)
        # print("Model vars:")
        # for p in model.get_global_vars(sess): print(p)

        # Check that all variables have been initialised:
        model.report_uninitialised(sess)
        # Perform fast adaptation:
        model.adapt(sess, data_set)
        # Retrieve the learned task-specific parameters:
        task_specific_params = model.get_global_vars(sess)
    
    # Have to reset the default graph, otherwise variables in this worker's
    # graph persist even after the session ends and those variables lose their
    # values and the worker starts handling a new task:
    tf.reset_default_graph()
    
    # Return the task specific parameters to the outer loop
    return task_specific_params
    # Should also return/log/plot a loss value?

def meta_train_parallel(model, meta_set, num_outer_steps=2000):

    with Pool() as pool, tf.Session() as sess:
        logging.info("Warming up the pool...")
        pool.map(int, [0])
        # # start_time = time()
        # model.initialise_variables(sess)
        # model.report_uninitialised(sess)

        # for epoch in range(num_outer_steps):
        #     logging.info("Epoch = {}".format(epoch))
        #     # Retrieve meta-parameters, ready to pass to the inner update:
        #     meta_params = model.get_global_vars(sess)
            
        #     inner_arg_list = [(meta_params, data_set) for data_set in task_set]
        #     # Each elem of the result is a list of task-specific params:
        #     task_specific_params_list = pool.starmap(
        #         inner_update, inner_arg_list
        #     )
            
        #     model.meta_update(sess, task_specific_params_list)
        
        # meta_params = model.get_global_vars(sess)
        # print("Final params:")
        # for p in meta_params: print(p)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    logging.info("Loading data...")
    meta_set = tasksets.SinusoidalMetaSet(tasksets.DEFAULT_FILE_NAME)
    
    logging.info("Creating model...")
    model = ReptileRegressor()

    meta_train_parallel(model, meta_set)