mpirun -np 5 python main_sync.py --algor fedavg --dataset mnist
mpirun -np 5 python main_sync.py --algor fedavg --dataset cifar10
mpirun -np 5 python main_sync.py --algor fedavg --dataset cifar100
