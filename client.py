from mpi4py import MPI
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from copy import deepcopy

from utils import load_data

from model import get_model, build_optimizer, build_lr_scheduler, build_loss_function


class ClientBase(object):
    def __init__(self, config, client_id):
        self.id = client_id
        self.model = get_model(
            config.model_name,
            in_channels=config.in_channels,
            num_classes=config.num_classes,
            img_size=config.img_size,
        )

        self.config = config
        self.train_data_loader = load_data(
            client_id=client_id,
            dataset_name=config.dataset,
            num_clients=config.num_clients,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            iid=config.iid,
            alpha=config.alpha,
            train=True,
        )
        self.num_samples = len(self.train_data_loader.dataset)

        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()

    def update_model(self, global_weights):
        self.model.load_state_dict(global_weights)

        self.optimizer = build_optimizer(
            self.config.optimizer_name,
            self.model.parameters(),
            self.config.optimizer_hyperparams,
        )
        self.scheduler = build_lr_scheduler(
            self.config.lr_scheduler_name,
            self.optimizer,
            self.config.lr_scheduler_hyperparams,
        )
        self.criterion = build_loss_function(self.config.loss_name)

        self.model.train()
        self.model.to(self.config.device)
        self.init_model = deepcopy(self.model.state_dict())
        local_weights = self.model.state_dict()

        for epoch in range(self.config.epochs):
            for idx, (inputs, labels) in enumerate(self.train_data_loader):
                inputs, labels = inputs.to(self.config.device), labels.to(
                    self.config.device
                )

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                if self.config.algor in ["fedprox"]:
                    l2_reg = self.l2_reg(self.model, self.init_model)
                    # logger.info(f"l2_reg: {self.config.lamda * l2_reg:.6f}")
                    loss += self.config.lamda * self.l2_reg(self.model, self.init_model)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                if idx % self.config.log_interval == 0:
                    logger.info(
                        f"Client {self.id} | "
                        f"Rank: {self.config.rank} | "
                        f"Epoch: {epoch}/{self.config.epochs} | "
                        f"Batch: {idx}/{len(self.train_data_loader)} | "
                        f"Loss: {loss.item():.6f}"
                    )

            self.scheduler.step()

        num_samples = len(self.train_data_loader.dataset)

        return local_weights, num_samples

    def get_weights(self):
        return self.model.state_dict()

    def set_weights(self, weights):
        self.model.load_state_dict(weights)

    def l2_reg(self, model, init_model):
        reg = 0
        for name, param in model.named_parameters():
            if "weight" in name:
                reg += torch.norm(param - init_model[name], 2)
        return reg


class ClientSync(ClientBase):
    def __init__(self, config, client_id):
        super().__init__(config, client_id)


class ClientAsync(ClientBase):
    def __init__(self, config, client_id):
        super().__init__(config, client_id)
