import numpy as np
import torch
import random
from torchvision import datasets, transforms
import os
from loguru import logger

def load_data(
    client_id: int = 0,
    dataset_name: str = "mnist",
    num_clients: int = 10,
    batch_size: int = 64,
    num_workers: int = 0,
    data_path=os.path.join(".", "data"),
    iid=True,
    alpha=0.1,
    train=True,
):
    if dataset_name == "mnist":
        # Load MNIST dataset
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        data = datasets.MNIST(
            data_path, train=train, download=True, transform=transform
        )
    elif dataset_name == "cifar10":
        # Load CIFAR-10 dataset
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )
        data = datasets.CIFAR10(
            data_path, train=train, download=True, transform=transform
        )
    elif dataset_name == "cifar100":
        # Load CIFAR-100 dataset
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
                ),
            ]
        )
        data = datasets.CIFAR100(
            data_path, train=train, download=True, transform=transform
        )
    else:
        raise ValueError(
            "Invalid dataset name. Allowed values are: mnist, cifar10, cifar100"
        )

    if not train:
        # Return test data
        return torch.utils.data.DataLoader(
            data, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    # Partition data among clients using non-iid strategy
    num_data = len(data)
    data_indices = list(range(num_data))
    random.shuffle(data_indices)
    data_indices_per_client = num_data // num_clients
    start = client_id * data_indices_per_client
    end = (client_id + 1) * data_indices_per_client

    if iid:
        # Partition data using an IID strategy
        data = torch.utils.data.Subset(data, data_indices[start:end])
    else:
        # Partition data using a non-IID strategy
        labels = np.array(data.targets)
        label_indices = [np.where(labels == i)[0] for i in range(10)]
        data_indices = []
        for i in range(num_clients):
            client_indices = []
            for j in range(int(alpha * 10)):
                label_index = i * int(alpha * 10) + j
                client_indices += list(label_indices[label_index])
            data_indices += client_indices

        data = torch.utils.data.Subset(data, data_indices)

    return torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    print("This is a utility file.")

    # Test the load_data function
    for dataset_name in ["mnist", "cifar10", "cifar100"]:
            
        train_loader = load_data(
            client_id=0,
            dataset_name=dataset_name,
            num_clients=10,
            batch_size=64,
            num_workers=0,
            data_path=os.path.join(".", "data"),
            iid=True,
            alpha=0.1,
            train=True,
        )

        test_loader = load_data(
            client_id=0,
            dataset_name=dataset_name,
            num_clients=10,
            batch_size=64,
            num_workers=0,
            data_path=os.path.join(".", "data"),
            iid=True,
            alpha=0.1,
            train=False,
        )

    