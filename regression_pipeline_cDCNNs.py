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

from network_architecture_cDCNNs import (ContractDeepConvolutionalNeuralNetworkSingleChannelsInitialBias,
                                  ContractDeepConvolutionalNeuralNetworkFullyConnectedSingleChannelsInitialBias )


class PyTorchDatasetConv1d(Dataset):

    def __init__(self, data_x, data_y):
        self.data_x = torch.from_numpy(data_x.astype("float32"))
        self.data_y = torch.from_numpy(data_y.astype("float32"))

    def __len__(self):
        return len(self.data_x)

    def __getitem__(self, idx):
        return self.data_x[idx], self.data_y[idx]



# cDCNN, single channel
class RegressionWithContractDeepConvolutionalNeuralNetworkSingleChannelsInitialBias(object):
    def __init__(self, X_train, Y_train, X_test, Y_test, num_layers=8,
                 filter_size=3, num_channels=1,bias_value=0.01,
                 learning_rate=0.01, gamma_value=0.9,
                 train_batch_size=200,
                 test_batch_size=100,
                 num_epoch=100,
                 scheduler_step=10,
                 gpu_id=0,
                 show_learning_process=False):

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.input_dim = int(self.X_train.shape[1])

        # params for cDCNN
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
        self.show_learning_process = show_learning_process

        if not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")

        seed = 123456
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        logger.info(f"self.X={self.X_train.shape}, self.Y={self.Y_train.shape}, self.input_dim={self.input_dim}  ")

        self.network = ContractDeepConvolutionalNeuralNetworkSingleChannelsInitialBias(input_dim=self.input_dim,
                                                                                       num_layers=self.num_layers,
                                                                                       in_channels=1,
                                                                                       out_channels=self.num_channels,
                                                                                       kernel_size=self.filter_size,
                                                                                       output_dim=1,
                                                                                       bias_value=bias_value).cuda(self.gpu_id)

        pytorch_total_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        logger.info("Total_params: {}".format(pytorch_total_params))
        logger.info(self.network)

        self.train_dataset = PyTorchDatasetConv1d(data_x=self.X_train, data_y=self.Y_train)
        self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=self.train_batch_size,
                                           num_workers=1, shuffle=True)

        self.test_dataset = PyTorchDatasetConv1d(data_x=self.X_test, data_y=self.Y_test)
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=self.test_batch_size, num_workers=1,
                                          shuffle=False)

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
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
                # logger.debug(f'input.shape={input.size()}')
                self.optimizer.zero_grad()
                output = self.network(input)
                loss = self.loss_function(output, target)
                loss.backward()
                self.optimizer.step()
                # logger.debug("training loss: {}".format(loss.cpu().item()))
                train_epoch_loss += loss.cpu().item()
                train_count += 1
            train_mse = train_epoch_loss / train_count

            if epoch % 50 == 0:
                logger.info("------------------ cDCNN single training at epoch={}, mse loss={}, RMSE={}, lr = {}".format(epoch, train_mse,
                                                                                                   np.sqrt(train_mse),
                                                                                                   self.scheduler.get_lr()))

            self.train_epoch_loss_list.append(train_epoch_loss / train_count)

            if epoch % self.scheduler_step == 0:
                self.scheduler.step()

            if epoch % 50 == 0:
                mse_value = self.testing()
                logger.info("inner cDCNN single testing mse loss: {}, RMSE = {}, at epoch={}".format(mse_value,
                                                                                                     np.sqrt(
                                                                                                         mse_value),
                                                                                                     epoch))

            if self.show_learning_process:
                if epoch % 10 == 0:
                    mse_value = self.testing()
                    logger.info("cDCNN single testing  at epoch={}, mse loss={}, RMSE={}".format(epoch, mse_value,
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


# cDCNN-fc, single channel
class RegressionWithContractDeepConvolutionalNeuralNetworkFullyConnectedSingleChannelsInitialBias(object):
    def __init__(self, X_train, Y_train, X_test, Y_test, num_layers=8, num_fc_layer=1,
                 filter_size=3, num_channels=1,  bias_value=0.01,
                 learning_rate=0.01, gamma_value=0.9,
                 train_batch_size=200, test_batch_size=100, num_epoch=100, scheduler_step=10,
                 gpu_id=0,
                 show_learning_process=False):

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.input_dim = int(self.X_train.shape[1])

        # params for cDCNN-fc
        self.num_layers = int(num_layers)
        self.num_fc_layer = int(num_fc_layer)
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
        self.show_learning_process = show_learning_process

        if not torch.cuda.is_available():
            raise Exception("No GPU found, please run without --cuda")

        seed = 123456
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        logger.info(f"self.X={self.X_train.shape}, self.Y={self.Y_train.shape}, self.input_dim={self.input_dim}  ")

        self.network = ContractDeepConvolutionalNeuralNetworkFullyConnectedSingleChannelsInitialBias(input_dim=self.input_dim,
                                                                                          num_layers=self.num_layers,
                                                                                          num_fc_layer=self.num_fc_layer,
                                                                                          in_channels=1,
                                                                                          out_channels=self.num_channels,
                                                                                          kernel_size=self.filter_size,
                                                                                          output_dim=1,
                                                                                          bias_value=bias_value).cuda(self.gpu_id)

        pytorch_total_params = sum(p.numel() for p in self.network.parameters() if p.requires_grad)
        logger.info("Total_params: {}".format(pytorch_total_params))
        logger.info(self.network)

        self.train_dataset = PyTorchDatasetConv1d(data_x=self.X_train, data_y=self.Y_train)
        self.train_dataloader = DataLoader(dataset=self.train_dataset, batch_size=self.train_batch_size,
                                           num_workers=1, shuffle=True)

        self.test_dataset = PyTorchDatasetConv1d(data_x=self.X_test, data_y=self.Y_test)
        self.test_dataloader = DataLoader(dataset=self.test_dataset, batch_size=self.test_batch_size, num_workers=1,
                                          shuffle=False)

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate, betas=(0.5, 0.999))
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
                # logger.debug(f'input.shape={input.size()}')
                self.optimizer.zero_grad()
                output = self.network(input)
                loss = self.loss_function(output, target)
                loss.backward()
                self.optimizer.step()
                # logger.debug("training loss: {}".format(loss.cpu().item()))
                train_epoch_loss += loss.cpu().item()
                train_count += 1
            train_mse = train_epoch_loss / train_count

            self.train_epoch_loss_list.append(train_epoch_loss / train_count)

            if epoch % 50 == 0:
                logger.info(
                    "------------------ cDCNN single training at epoch={}, mse loss={}, RMSE={}, lr = {}".format(epoch,
                                                                                                                 train_mse,
                                                                                                                 np.sqrt(
                                                                                                                     train_mse),
                                                                                                                 self.scheduler.get_lr()))

            if epoch % self.scheduler_step == 0:
                self.scheduler.step()

            if epoch % 50 == 0:
                mse_value = self.testing()
                logger.info("inner cDCNN-fc single testing mse loss: {}, RMSE = {}, at epoch={}".format(mse_value,
                                                                                                        np.sqrt(
                                                                                                            mse_value),
                                                                                                        epoch))

            if self.show_learning_process:
                if epoch % 10 == 0:
                    mse_value = self.testing()
                    logger.info("cDCNN-fc single testing  at epoch={}, mse loss={}, RMSE={}".format(epoch, mse_value,
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

