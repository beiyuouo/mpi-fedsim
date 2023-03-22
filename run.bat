mpiexec -n 5 python main_sync.py --algor fedavg --dataset mnist
mpiexec -n 5 python main_sync.py --algor fedavg --dataset cifar10
mpiexec -n 5 python main_sync.py --algor fedavg --dataset cifar100
