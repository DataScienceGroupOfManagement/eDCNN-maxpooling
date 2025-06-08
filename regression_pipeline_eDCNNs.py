import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingWarmRestarts, CyclicLR
from torchvision import transforms
import torch.nn.functional as F
import datetime
from scipy.stats import qmc
from torch.utils.data import Dataset
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np

from network_architecture_eDCNNs import eDCNN_maxpooling_network, eDCNN_zeropadding_network


class PyTorchDatasetConv1dMaxpooling(Dataset):

    def __init__(self, data_x, data_y, gpu_id):
        self.data_x = torch.from_numpy(data_x.astype("float32"))
        self.data_y = torch.from_numpy(data_y.astype("float32"))

        self.data_x.cuda(gpu_id)
        self.data_y.cuda(gpu_id)

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]


class regression_with_eDCNN_maxpooling(object):
    def __init__(self, X_train, Y_train, X_test, Y_test,
                 num_layers=5, filter_size=5, num_channels=1, pooling_size=2, bias_value=0.01,
                 learning_rate=0.01, gamma_value=0.9,
                 train_batch_size=200,
                 test_batch_size=100,
                 num_epoch=100,
                 scheduler_step=10,
                 gpu_id=0,
                 optimizer_name='Adam',
                 momentum_value = 0.0,  # momentum_value for SGD
                 show_learning_process=False):

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

        self.input_dim = int(self.X_train.shape[1])

        # params for eDCNN-maxpooling
        self.num_layers = int(num_layers)
        self.filter_size = int(filter_size)
        self.num_channels = int(num_channels)
        self.pooling_size = int(pooling_size)
        self.bias_value = bias_value

        # params for training
        self.learning_rate = learning_rate
        self.gamma_value = gamma_value
        self.train_batch_size = int(train_batch_size)
        self.test_batch_size = int(test_batch_size)
        self.num_epoch = int(num_epoch)
        self.scheduler_step = int(scheduler_step)
        self.gpu_id = int(gpu_id)
        self.optimizer_name = optimizer_name
        self.momentum_value = momentum_value # for SGD
        self.show_learning_process = show_learning_process

        if not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")

        seed = 123456
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        logger.info(f"self.X={self.X_train.shape}, self.Y={self.Y_train.shape}, self.input_dim={self.input_dim}  ")

        self.network = eDCNN_maxpooling_network(
                input_dim=self.input_dim,
                num_layers=self.num_layers,
                in_channels=1,
                out_channels=self.num_channels,
                kernel_size=self.filter_size,
                pooling_size=self.pooling_size,
                bias_value=self.bias_value,
                output_dim=1).cuda(self.gpu_id)

        pytorch_total_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        logger.info("Total_params: {}".format(pytorch_total_params))
        logger.info(self.network)

        self.train_dataset = PyTorchDatasetConv1dMaxpooling(data_x=self.X_train, data_y=self.Y_train,
                                                            gpu_id=self.gpu_id)
        self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=self.train_batch_size,
                                           num_workers=1, shuffle=True)

        self.test_dataset = PyTorchDatasetConv1dMaxpooling(data_x=self.X_test, data_y=self.Y_test, gpu_id=self.gpu_id)
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=self.test_batch_size, num_workers=1,
                                          shuffle=False)

        if self.optimizer_name == 'Adam':
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
            logger.info('using optimizer Adam !')
        elif self.optimizer_name == 'SGD':
            self.optimizer = optim.SGD(self.network.parameters(),
                                       lr=self.learning_rate,
                                       momentum=self.momentum_value,
                                       weight_decay=1e-4,
                                       nesterov=True)
            logger.info('using optimizer SGD !')
        elif self.optimizer_name == 'RMSprop':
            self.optimizer = optim.RMSprop(self.network.parameters(), lr=self.learning_rate, )
            logger.info('using optimizer RMSprop !')
        elif self.optimizer_name == 'Adagrad':
            self.optimizer = optim.Adagrad(self.network.parameters(), lr=self.learning_rate, )
            logger.info('using optimizer Adagrad !')
        elif self.optimizer_name == 'Adadelta':
            self.optimizer = optim.Adadelta(self.network.parameters(), lr=self.learning_rate,  weight_decay=1e-4,)
            logger.info('using optimizer Adadelta !')
        else:
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
            logger.info('using default optimizer Adam !')

        self.scheduler = ExponentialLR(self.optimizer, gamma=self.gamma_value)
        self.loss_function = nn.MSELoss().cuda(self.gpu_id)

    def training(self):

        self.train_epoch_loss_list = []
        self.test_epoch_loss_list = []

        for epoch in range(1, self.num_epoch + 1):
            train_epoch_loss = 0.0
            train_count = 0.0
            for iter_idx, inputs in enumerate(self.train_dataloader, 1):
                input, target = Variable(inputs[0]).cuda(self.gpu_id), Variable(inputs[1]).cuda(self.gpu_id)
                input = input.unsqueeze(1)
                self.optimizer.zero_grad()
                output = self.network(input)
                loss = self.loss_function(output, target)
                loss.backward()
                self.optimizer.step()
                # logger.debug("training loss: {}".format(loss.cpu().item()))
                train_epoch_loss += loss.cpu().item()
                train_count += 1
            train_mse = train_epoch_loss / train_count
            # logger.info("eDCNN-maxpooling-single training at epoch={}, mse loss={}, RMSE={}, lr = {}".format(epoch, train_mse,
            #                                                                                        np.sqrt(train_mse),
            #                                                                                        self.scheduler.get_lr()))
            self.train_epoch_loss_list.append(train_mse)

            if epoch % self.scheduler_step == 0:
                self.scheduler.step()

            if epoch % 50 == 0:
                mse_value = self.testing()
                logger.info(
                    "inner eDCNN-maxpooling-single testing mse loss: {}, RMSE = {}, at epoch={}".format(mse_value,
                                                                                                        np.sqrt( mse_value),
                                                                                                        epoch))

            if self.show_learning_process:
                if epoch % 10 == 0:
                    mse_value = self.testing()
                    logger.info(
                        "eDCNN-maxpooling-single testing  at epoch={}, mse loss={}, RMSE={}".format(epoch, mse_value,
                                                                                                    np.sqrt(mse_value)))
                    self.test_epoch_loss_list.append(mse_value)

        if self.show_learning_process:
            return self.train_epoch_loss_list, self.test_epoch_loss_list
        else:
            return self.train_epoch_loss_list

    def testing(self):
        test_epoch_loss = 0.0
        test_count = 0.0
        for iter_idx, inputs in enumerate(self.test_dataloader, 1):
            with torch.no_grad():
                input, target = Variable(inputs[0]).cuda(self.gpu_id), Variable(inputs[1]).cuda(self.gpu_id)
                input = input.unsqueeze(1)
                output = self.network(input)
                loss = self.loss_function(output, target)
                test_epoch_loss += loss.cpu().item()
                test_count += 1
        test_mse = test_epoch_loss / test_count
        return test_mse


class regression_with_eDCNN_zeropadding(object):
    def __init__(self, X_train, Y_train, X_test, Y_test,
                 num_layers=5,
                 filter_size=5,
                 num_channels=1,
                 bias_value=0.01,
                 learning_rate=0.01,
                 gamma_value=0.9,
                 train_batch_size=200,
                 test_batch_size=100,
                 num_epoch=100,
                 scheduler_step=10,
                 gpu_id=0,
                 optimizer_name='Adam',
                 show_learning_process=False):
        logger.info('regression_with_eDCNN_zeropadding starting !')

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

        self.input_dim = int(self.X_train.shape[1])

        # params for eDCNN-maxpooling zero-padding
        self.num_layers = int(num_layers)
        self.filter_size = int(filter_size)
        self.num_channels = int(num_channels)
        self.bias_value = bias_value

        # params for training
        self.learning_rate = learning_rate
        self.gamma_value = gamma_value
        self.train_batch_size = int(train_batch_size)
        self.test_batch_size = int(test_batch_size)
        self.num_epoch = int(num_epoch)
        self.scheduler_step = int(scheduler_step)
        self.gpu_id = int(gpu_id)
        self.optimizer_name = optimizer_name
        self.show_learning_process = show_learning_process

        if not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")

        seed = 123456
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        logger.info(f"self.X={self.X_train.shape}, self.Y={self.Y_train.shape}, self.input_dim={self.input_dim}  ")

        self.network = eDCNN_zeropadding_network(
                input_dim=self.input_dim,
                num_layers=self.num_layers,
                in_channels=1,
                out_channels=self.num_channels,
                kernel_size=self.filter_size,
                bias_value=self.bias_value,
                output_dim=1).cuda(self.gpu_id)

        pytorch_total_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        logger.info("Total_params: {}".format(pytorch_total_params))
        logger.info(self.network)

        self.train_dataset = PyTorchDatasetConv1dMaxpooling(data_x=self.X_train, data_y=self.Y_train,
                                                            gpu_id=self.gpu_id)
        self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=self.train_batch_size,
                                           num_workers=1, shuffle=True)

        self.test_dataset = PyTorchDatasetConv1dMaxpooling(data_x=self.X_test, data_y=self.Y_test, gpu_id=self.gpu_id)
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=self.test_batch_size, num_workers=1,
                                          shuffle=False)

        if self.optimizer_name == 'Adam':
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
            logger.info('using optimizer Adam !')
        else:
            self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
            logger.info('using default optimizer Adam !')

        self.scheduler = ExponentialLR(self.optimizer, gamma=self.gamma_value)
        self.loss_function = nn.MSELoss().cuda(self.gpu_id)

    def training(self):

        self.train_epoch_loss_list = []
        self.test_epoch_loss_list = []

        for epoch in range(1, self.num_epoch + 1):
            train_epoch_loss = 0.0
            train_count = 0.0
            for iter_idx, inputs in enumerate(self.train_dataloader, 1):
                input, target = Variable(inputs[0]).cuda(self.gpu_id), Variable(inputs[1]).cuda(self.gpu_id)
                input = input.unsqueeze(1)
                self.optimizer.zero_grad()
                output = self.network(input)
                loss = self.loss_function(output, target)
                loss.backward()
                self.optimizer.step()
                # logger.debug("training loss: {}".format(loss.cpu().item()))
                train_epoch_loss += loss.cpu().item()
                train_count += 1
            train_mse = train_epoch_loss / train_count
            # logger.info("eDCNN-maxpooling-single training at epoch={}, mse loss={}, RMSE={}, lr = {}".format(epoch, train_mse,
            #                                                                                        np.sqrt(train_mse),
            #                                                                                        self.scheduler.get_lr()))
            self.train_epoch_loss_list.append(train_mse)

            if epoch % self.scheduler_step == 0:
                self.scheduler.step()

            if epoch % 50 == 0:
                mse_value = self.testing()
                logger.info(
                    "inner eDCNN-zeropadding-single testing mse loss: {}, RMSE = {}, at epoch={}".format(mse_value,
                                                                                                        np.sqrt( mse_value),
                                                                                                        epoch))

            if self.show_learning_process:
                if epoch % 10 == 0:
                    mse_value = self.testing()
                    logger.info(
                        "eDCNN-zeropadding-single testing  at epoch={}, mse loss={}, RMSE={}".format(epoch, mse_value,
                                                                                                    np.sqrt(mse_value)))
                    self.test_epoch_loss_list.append(mse_value)

        if self.show_learning_process:
            return self.train_epoch_loss_list, self.test_epoch_loss_list
        else:
            return self.train_epoch_loss_list

    def testing(self):
        test_epoch_loss = 0.0
        test_count = 0.0
        for iter_idx, inputs in enumerate(self.test_dataloader, 1):
            with torch.no_grad():
                input, target = Variable(inputs[0]).cuda(self.gpu_id), Variable(inputs[1]).cuda(self.gpu_id)
                input = input.unsqueeze(1)
                output = self.network(input)
                loss = self.loss_function(output, target)
                test_epoch_loss += loss.cpu().item()
                test_count += 1
        test_mse = test_epoch_loss / test_count
        return test_mse

