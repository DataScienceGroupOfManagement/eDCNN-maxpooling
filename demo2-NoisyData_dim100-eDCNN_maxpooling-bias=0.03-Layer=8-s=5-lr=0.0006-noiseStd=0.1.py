from __future__ import print_function
import argparse
from math import log10
import numpy as np
from loguru import logger
from tqdm import tqdm

# simulation functions
from simulation_functions import smooth_dim100_noisy_data
from simulation_functions import smooth_dim100_noise_free_data

from regression_pipeline_eDCNNs import  regression_with_eDCNN_maxpooling

if __name__ == '__main__':

    # target function
    train_sample_size = 2000
    test_sample_size = 500
    data_dim = 100
    repeated_times = 10

    # the noise level
    noise_std = 0.1
    noise_variance = noise_std * noise_std

    seed = 2025
    np.random.seed(seed)

    # settings for eDCNN-maxpooling
    num_layers = 8
    filter_size = 5
    num_channels = 1  # 单通道的网络结构
    pooling_size = 2

    bias_value = 0.03   # 给定值；
    learning_rate = 0.0006

    # settings for training
    gamma_value = 0.95
    train_batch_size = 200
    test_batch_size = test_sample_size

    num_epoch = 4000
    scheduler_step = 400

    gpu_id = 0

    # ----------------------------------------------------------------
    import os

    # add logs
    fig_save_path = './newlogs_noisydata/eDCNN-maxpooling-smoothDim100/'
    if not os.path.exists(fig_save_path): os.makedirs(fig_save_path)
    file_name = (
        f'eDCNN-maxpoloing_Ntr={train_sample_size}_Nte={test_sample_size}_num_layers={num_layers}_filter_size={filter_size}_channels={num_channels}'
        f'_pooling_size={pooling_size}_epoch={num_epoch}_b={bias_value}_lr={learning_rate}_noiseStd={noise_std}')

    logger.add(fig_save_path + f"{file_name}.log", enqueue=True)
    # ----------------------------------------------------------------
    logger.info(file_name)

    bias_value_list = [ bias_value ]
    learning_rate_list = [learning_rate]

    test_avgRMSE_matrix = np.zeros((len(bias_value_list), len(learning_rate_list)))

    for row_idx in range(len(bias_value_list)):
        for col_idx in range(len(learning_rate_list)):

            bias_value = bias_value_list[row_idx]
            learning_rate = learning_rate_list[col_idx]

            train_mse_list = []
            train_rmse_list = []

            test_mse_list = []
            test_rmse_list = []

            for ex_iter in tqdm(range(repeated_times)):
                logger.info(
                    f'*******bias_value={bias_value},  learning_rate={learning_rate}********ex_iter={ex_iter}****************************')

                x_train, y_train = smooth_dim100_noisy_data(sample_size=train_sample_size, data_dim=data_dim,
                                                            noise_variance=noise_variance)  # noisy train data

                x_test, y_test = smooth_dim100_noise_free_data(sample_size=test_sample_size,
                                                               data_dim=data_dim)  # noise free test data

                eDCNN_maxpooling_network = regression_with_eDCNN_maxpooling(
                    X_train=x_train,
                    Y_train=y_train,
                    X_test=x_test,
                    Y_test=y_test,
                    num_layers=num_layers,
                    filter_size=filter_size,
                    num_channels=num_channels,
                    pooling_size=pooling_size,
                    bias_value=bias_value,
                    learning_rate=learning_rate,
                    gamma_value=gamma_value,
                    train_batch_size=train_batch_size,
                    test_batch_size=test_batch_size,
                    num_epoch=num_epoch,
                    scheduler_step=scheduler_step,
                    gpu_id=gpu_id)

                train_epoch_loss_list = eDCNN_maxpooling_network.training()
                train_mse = train_epoch_loss_list[-1]
                train_mse_list.append(train_mse)
                train_rmse_list.append(np.sqrt(train_mse))

                test_mse = eDCNN_maxpooling_network.testing()
                logger.info(
                    f'Train MSE ={train_mse}, RMSE = {np.sqrt(train_mse)}; \t Test MSE = {test_mse}, '
                    f'RMSE = {np.sqrt(test_mse)} at func dim={data_dim} !')

                test_mse_list.append(test_mse)
                test_rmse_list.append(np.sqrt(test_mse))

            # ----------------------------------------------
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

            test_avgRMSE_matrix[row_idx, col_idx] = avg_test_rmse
            logger.info(
                f'presented results: bias_value={bias_value},  learning_rate={learning_rate}, avg_test_rmse = {avg_test_rmse} ')

    logger.info(f"bias_value_list = {bias_value_list}")
    logger.info(f"learning_rate_list = {learning_rate_list}")
    logger.info(f'test_avgRMSE_matrix={test_avgRMSE_matrix}')
