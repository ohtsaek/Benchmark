# Model or training components
from argparse import Namespace
import torch
from torch import nn as nn
import torch.nn.functional as F

def get_activation_function(activation: str) -> nn.Module:
    """
    Gets an activation function module given the name of the activation.

    :param activation: The name of the activation function.
    :return: The activation function module.
    """
    if activation == 'ReLU':
        return nn.ReLU()
    elif activation == 'LeakyReLU':
        return nn.LeakyReLU(0.1)
    elif activation == 'PReLU':
        return nn.PReLU()
    elif activation == 'tanh':
        return nn.Tanh()
    elif activation == 'SELU':
        return nn.SELU()
    elif activation == 'ELU':
        return nn.ELU()
    elif activation == "Linear":
        return lambda x: x
    else:
        raise ValueError(f'Activation "{activation}" not supported.')



def get_loss_func(args: Namespace):
    """
    Gets the loss function corresponding to a given dataset type.

    :param args: Namespace containing the dataset type ("classification" or "regression").
    :return: A PyTorch loss function.
    """

    if args.dataset_type == 'classification':
        loss_func = nn.BCEWithLogitsLoss(reduction='none')
    elif args.dataset_type == 'regression':
        loss_func = nn.MSELoss(reduction='mean')
    return loss_func


def build_optimizer(model: nn.Module, args: Namespace):
    """
    Builds an Optimizer.

    :param model: The model to optimize.
    :param args: Arguments.
    :return: An initialized Optimizer.
    """
    return torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.init_lr, weight_decay=args.weight_decay)  







