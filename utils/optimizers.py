import torch
from muon import SingleDeviceMuonWithAuxAdam


def create_muon_optimizer(lr):
    """
    Returns a function mk_optim(model) that creates a MuonWithAuxAdam optimizer
    with the correct parameter groups.
    """

    def mk_optim(params):
        """
        Returns MuonWithAuxAdam optimizer with the specified learning rate.
        """
        param_list = list(params)
        hidden_weights = [p for p in param_list if p.ndim >= 2]
        others = [p for p in param_list if p.ndim < 2]

        param_groups = [
            dict(params=hidden_weights, use_muon=True, lr=lr, weight_decay=lr / 1e3),
            dict(params=others, use_muon=False, lr=lr, weight_decay=lr / 1e3),
        ]

        return SingleDeviceMuonWithAuxAdam(param_groups)

    return mk_optim


def create_adam_optimizer(lr):
    """
    Create a function that returns Adam optimizer with a specific learning rate.

    Args:
        lr (float): Learning rate for the optimizer.

    Returns:
        function: Function that generates Adam optimizer with the specified learning rate.
    """

    def mk_optim(params):
        """
        Returns Adam optimizer with the specified learning rate.

        Args:
            params: Model parameters.

        Returns:
            torch.optim.Adam: Adam optimizer.
        """
        return torch.optim.Adam(params, lr=lr, weight_decay=lr / 1e3)

    return mk_optim
