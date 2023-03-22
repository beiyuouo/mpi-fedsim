from .fedavg import fedavg


def get_algor(algor_name):
    if algor_name == "fedavg":
        return fedavg
    elif algor_name == "fedprox":
        return fedavg
    else:
        raise ValueError(f"Unknown algor name: {algor_name}")


def require_num_samples(cfg):
    if cfg.algor in ["fedbuff"] or cfg.sync:
        return True
    return False
