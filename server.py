from mpi4py import MPI
import numpy as np
import random
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from loguru import logger

from model import get_model, build_loss_function
from utils import load_data
from algor import get_algor, require_num_samples


class ServerBase(object):
    def __init__(self, config, model):
        self.model = model.to(config.device)
        self.criterion = build_loss_function(config.loss_name)

        self.config = config
        self.test_data_loader = load_data(
            dataset_name=config.dataset,
            batch_size=config.test_batch_size,
            num_workers=config.num_workers,
            train=False,
        )

        self.global_weights = self.model.state_dict()
        self.updates_received = 0
        self.agg = get_algor(self.config.algor)

        log_path = os.path.join(
            self.config.log_dir, self.config.dataset, self.config.algor
        )

        if self.config.iid:
            log_path = os.path.join(
                self.config.log_dir, self.config.dataset, self.config.algor
            )
        else:
            log_path = os.path.join(
                self.config.log_dir,
                f"{self.config.dataset}_non-iid",
                self.config.algor,
            )

        SummaryWriter(log_dir=log_path)
        self.writer = SummaryWriter(log_dir=log_path)

        self.current_round = 0

        self.comm = MPI.COMM_WORLD
        self.size = self.comm.Get_size()
        self.rank = self.comm.Get_rank()
        self.clients = list(range(self.config.num_clients))

    def select_clients(self):
        return random.sample(self.clients, self.config.num_select_clients_per_round)

    def evaluate(self):
        self.model.eval()
        running_loss = 0.0
        correct = 0

        with torch.no_grad():
            for inputs, labels in self.test_data_loader:
                inputs, labels = inputs.to(self.config.device), labels.to(
                    self.config.device
                )
                outputs = self.model(inputs)

                running_loss += self.criterion(outputs, labels).item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                correct += preds.eq(labels).sum().item()

        test_loss = running_loss / len(self.test_data_loader.dataset)
        test_accuracy = 100.0 * correct / len(self.test_data_loader.dataset)

        logger.info(
            "Test loss: {:.6f}, Test accuracy: {:.2f}%".format(test_loss, test_accuracy)
        )

        self.writer.add_scalar("Test loss", test_loss, self.current_itr)
        self.writer.add_scalar("Test accuracy", test_accuracy, self.current_itr)

        return test_loss, test_accuracy


class ServerSync(ServerBase):
    def __init__(self, config, model):
        super().__init__(config, model)

        self.clients_num_samples = np.zeros(config.num_clients, dtype=np.int32)
        self.clients_local_weights = np.zeros(
            config.num_clients, dtype=object
        )  # object is a placeholder for any type

    def run(self):
        client_num_samples = []
        client_ids = []
        for j in range(1, self.size):
            client_num_samples, client_ids = self.comm.recv(source=j)
            self.clients_num_samples[client_ids] = client_num_samples
            logger.info(f"Received {client_num_samples} from client {client_ids}")

        logger.info(f"Received {self.clients_num_samples} from all clients")

        for i in range(self.config.num_rounds):
            logger.info(f"starting round {i + 1} of {self.config.num_rounds}")

            self.current_round = i + 1
            # Select clients for this round
            selected_clients = self.select_clients()

            # Train on selected clients' data
            global_weights = self.model.state_dict()

            client_weights = []

            logger.info(f"broadcasting global weights to all clients")

            for j in range(1, self.size):
                self.comm.send(global_weights, dest=j)

            client_ids = []

            for j in range(1, self.size):
                client_weights_, client_ids_ = self.comm.recv(source=j)
                if client_weights_ is not None:
                    client_weights.extend(client_weights_)
                    client_ids.extend(client_ids_)

                    logger.info(
                        f"reveived weights of client {client_ids_} from worker {j}"
                    )

            global_weights = self.agg(
                global_weights,
                client_weights,
                client_ids,
                selected_clients,
                clients_num_samples=self.clients_num_samples,
            )
            self.model.load_state_dict(global_weights)

            # Evaluate on test data and print metrics
            self.current_itr = self.current_round * self.config.num_clients
            test_loss, test_accuracy = self.evaluate()

        for j in range(1, self.size):
            # set timeout to 1 second
            self.comm.send("done", dest=j)


class ServerAsync(ServerBase):
    def __init__(self, config, model):
        super().__init__(config, model)

        self.client_version = np.zeros(config.num_clients, dtype=np.int32)
        self.client_num_samples = {}

    def run(self):
        if require_num_samples(self.config):
            for i in range(1, self.size):
                client_num_samples, client_ids = self.comm.recv(source=i)
                self.client_num_samples[client_ids] = client_num_samples

        for i in range(1, self.size):
            self.comm.send(self.global_weights, dest=i)
            self.client_version[i - 1] = 0

        while self.updates_received < self.config.num_rounds * self.config.num_clients:
            # Wait for an update from any client
            status = MPI.Status()
            local_model, num_sample, client_id = self.comm.recv(
                source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status
            )

            logger.info(f"Received update from client {client_id}")

            self.client_num_samples[client_id] = num_sample

            source = status.Get_source()

            # Update the global weights with the received update
            self.global_weights = self.agg(
                self.global_weights,
                local_model,
                source=client_id,
                version=self.updates_received,
                cfg=self.config,
                client_num_samples=self.client_num_samples,
            )

            # Load the updated global weights into the model
            self.model.load_state_dict(self.global_weights)

            # Send updated global weights to the client that sent the latest update
            self.comm.send(self.global_weights, dest=source)

            self.updates_received += 1
            self.client_version[client_id] = self.updates_received

            if self.updates_received % self.config.num_clients == 0:
                self.current_round += 1
                self.current_itr = self.updates_received
                test_loss, test_accuracy = self.evaluate()

        count_done = 0
        while True:
            status = MPI.Status()
            self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
            source = status.Get_source()
            self.comm.send("done", dest=source)
            count_done += 1

            if count_done == self.size - 1:
                break
