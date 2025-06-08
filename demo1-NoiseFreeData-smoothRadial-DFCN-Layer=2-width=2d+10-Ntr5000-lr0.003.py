from __future__ import print_function
import argparse
from math import log10
import numpy as np
from loguru import logger
from tqdm import tqdm

# simulation functions
from simulation_functions import smooth_radial_func_noise_free_data
from regression_pipeline_DFCN import RegressionWithDeepFullyConnectedNetwork

if __name__ == '__main__':

    # target function
    # def smooth_radial_func_noise_free_data(sample_size,   data_dim, ):
    train_sample_size = 5000
    test_sample_size = 1000
    data_dim = 4

    # experimental settings
    repeated_times = 10

    seed = 2025
    np.random.seed(seed)

    # settings for DFCN
    num_layers = 2
    hidden_dim = int(2 * data_dim + 10)  # network with fixed width = 2d+10

    # settings for training
    learning_rate = 0.003
    gamma_value = 0.95
    batch_size = 200
    test_batch_size = test_sample_size
    num_epoch = 4000
    scheduler_step = 400

    gpu_id = 0

    # ----------------------------------------------------------------
    # add logs
    import os
    fig_save_path = './newlogs/DFCN-smoothRadial-dim4/'
    if not os.path.exists(fig_save_path): os.makedirs(fig_save_path)
    file_name = ( f'DFCN_num_layers={num_layers}_hidden_dim={hidden_dim}'
        f'_num_epoch={num_epoch}_lr={learning_rate}_times={repeated_times}')
    logger.add(fig_save_path + f"{file_name}.log", enqueue=True)
    # ----------------------------------------------------------------
    logger.info(file_name)

    train_mse_list = []
    train_rmse_list = []

    test_mse_list = []
    test_rmse_list = []

    for ex_iter in tqdm(range(repeated_times)):
        logger.info(f'*******learning_rate={learning_rate}********ex_iter={ex_iter}****************************')

        x_train, y_train = smooth_radial_func_noise_free_data(sample_size=train_sample_size, data_dim=data_dim)
        x_test, y_test = smooth_radial_func_noise_free_data(sample_size=test_sample_size, data_dim=data_dim)

        DFCN_network = RegressionWithDeepFullyConnectedNetwork(
            X_train=x_train,
            Y_train=y_train,
            X_test=x_test,
            Y_test=y_test,
            num_layers=num_layers, hidden_dim=hidden_dim,
            learning_rate=learning_rate, gamma_value=gamma_value,
            train_batch_size=batch_size,
            test_batch_size=test_batch_size,
            num_epoch=num_epoch,
            scheduler_step=scheduler_step,
            gpu_id=gpu_id)

        train_epoch_loss_list = DFCN_network.training()

        train_mse = train_epoch_loss_list[-1]
        train_mse_list.append(train_mse)
        train_rmse_list.append(np.sqrt(train_mse))

        test_mse = DFCN_network.testing()
        logger.info(
            f'Train MSE ={train_mse}, RMSE = {np.sqrt(train_mse)}; \t Test MSE = {test_mse}, RMSE = {np.sqrt(test_mse)} at func dim={data_dim} !')

        test_mse_list.append(test_mse)
        test_rmse_list.append(np.sqrt(test_mse))

    # train results
    logger.info(f"train_mse_list = {train_mse_list}")
    avg_train_mse = np.mean(train_mse_list)
    logger.info(f'avg_train_mse ={avg_train_mse}')

    logger.info(f"train_rmse_list = {train_rmse_list}")
    avg_train_rmse = np.mean(train_rmse_list)
    logger.info(f'avg_train_rmse ={avg_train_rmse}')

    # ----------------------------------------------
    # test results
    logger.info(f"test_mse_list = {test_mse_list}")
    avg_test_mse = np.mean(test_mse_list)
    logger.info(f'avg_test_mse={avg_test_mse}')

    logger.info(f"test_rmse_list = {test_rmse_list}")
    avg_test_rmse = np.mean(test_rmse_list)
    logger.info(f'avg_test_rmse={avg_test_rmse}')

    logger.info(f'*********************results-> learning rate = {learning_rate}, avg_test_rmse={avg_test_rmse}')
