import torch
from muon import (
    SingleDeviceMuonWithAuxAdam,
    SingleDeviceMuonWithAuxAdamSpectralNorm,
    SingleDeviceMuonWithAuxAdam10p2,
    Adam10p,
)


def map_create_optimizer(optimizer_name):
    if optimizer_name == "adam":
        return create_adam_optimizer
    elif optimizer_name == "muon":
        return create_muon_optimizer
    elif optimizer_name == "10p":
        return create_10p_optimizer
    elif optimizer_name == "muon10p":
        return create_muon_10p_optimizer
    elif optimizer_name == "muonspectralnorm":
        return create_muon_spectralnorm_optimizer
    elif optimizer_name == "adamw":
        return create_adamw_optimizer
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def create_muon_spectralnorm_optimizer(lr):
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

        return SingleDeviceMuonWithAuxAdamSpectralNorm(param_groups)

    return mk_optim


def create_muon_10p_optimizer(lr):
    """
    Returns a function mk_optim(model) that creates a MuonWithAuxAdam10p optimizer
    with the correct parameter groups.
    """

    def mk_optim(params):
        """
        Returns MuonWithAuxAdam10p optimizer with the specified learning rate.
        """
        param_list = list(params)
        hidden_weights = [p for p in param_list if p.ndim >= 2]
        others = [p for p in param_list if p.ndim < 2]

        param_groups = [
            dict(params=hidden_weights, use_muon=True, lr=lr, weight_decay=lr / 1e3),
            dict(params=others, use_muon=False, lr=lr, weight_decay=lr / 1e3),
        ]

        return SingleDeviceMuonWithAuxAdam10p2(param_groups)

    return mk_optim


def create_10p_optimizer(lr):
    """
    Returns a function mk_optim(model) that creates a MuonWithAuxAdam10p optimizer
    with the correct parameter groups.
    """

    def mk_optim(params):
        """
        Returns MuonWithAuxAdam10p optimizer with the specified learning rate.
        """
        param_list = list(params)
        hidden_weights = [p for p in param_list if p.ndim >= 2]
        others = [p for p in param_list if p.ndim < 2]

        param_groups = [
            dict(params=hidden_weights, use_muon=True, lr=lr, weight_decay=lr / 1e3),
            dict(params=others, use_muon=False, lr=lr, weight_decay=lr / 1e3),
        ]

        return Adam10p(param_groups)

    return mk_optim


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


def create_adamw_optimizer(lr):
    """
    Create a function that returns AdamW optimizer with a specific learning rate.

    Args:
        lr (float): Learning rate for the optimizer.

    Returns:
        function: Function that generates AdamW optimizer with the specified learning rate.
    """

    def mk_optim(params):
        """
        Returns AdamW optimizer with the specified learning rate.

        Args:
            params: Model parameters.

        Returns:
            torch.optim.AdamW: AdamW optimizer.
        """
        return torch.optim.AdamW(params, lr=lr, weight_decay=lr / 1e3)

    return mk_optim
