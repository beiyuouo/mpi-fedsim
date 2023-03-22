import torch
from loguru import logger


def fedavg(
    global_weight, local_weights, client_ids, selected_client_ids, clients_num_samples
):
    """Federated averaging aggregation function.

    Args:
        global_weight (dict): Global model weights.
        local_weights (list): List of local model weights.
        clients_num_samples (list): List of number of samples for each client.

    Returns:
        dict: Global model weights.
    """

    global_weight = global_weight.copy()
    for key in global_weight.keys():
        global_weight[key] = torch.zeros_like(global_weight[key])

    agg_weight = {client_id: 0 for client_id in client_ids}

    for i, client_id in enumerate(selected_client_ids):
        agg_weight[client_id] = clients_num_samples[client_id]

    sum_agg_weight = sum(agg_weight.values())

    for key in agg_weight.keys():
        agg_weight[key] = agg_weight[key] / sum_agg_weight

    logger.info(f"agg_weight: {agg_weight}")

    for local_weight, client_id in zip(local_weights, client_ids):
        for key in global_weight.keys():
            global_weight[key] += agg_weight[client_id] * local_weight[key]

    return global_weight
