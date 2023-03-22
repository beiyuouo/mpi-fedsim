# mpi federated learning simulation

Accelerating Federated Learning Simulations Using MPI

## Introduction

See the [blog](https://blog.bj-yan.top/p/blog-accelerating-federated-learning-simulation-using-mpi/) for more details.

## Requirements

- Python 3.8+
- PyTorch 1.8+

## Usage

```bash
# install requirements
pip install -r requirements.txt

# dataset prepare
python utils.py  # check and download the dataset you need

# run the simulation
# for linux
mpirun -np 4 python main_sync.py  # 4 is the number of processes
# for windows
mpiexec -n 4 python main_sync.py  # 4 is the number of processes

# launch the tensorboard
tensorboard --logdir=logs
```
