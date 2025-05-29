import torch
from muon import (
    SingleDeviceMuonWithAuxAdam,
    SingleDeviceMuonWithAuxAdamSpectralNorm,
    SingleDeviceMuonWithAuxAdam10p2,
    Adam10p,
    AdamWSpectralNorm,
)


def split_params(params):
    param_list = list(params)
    hidden_weights = [p for p in param_list if p.ndim >= 2]
    others = [p for p in param_list if p.ndim < 2]
    return hidden_weights, others


def muon_param_group_factory(optim_cls, use_muon_for_hidden=True):
    def factory(lr):
        def mk_optim(params):
            hidden_weights, others = split_params(params)
            param_groups = [
                dict(
                    params=hidden_weights,
                    use_muon=use_muon_for_hidden,
                    lr=lr,
                    weight_decay=lr / 1e3,
                ),
                dict(params=others, use_muon=False, lr=lr, weight_decay=lr / 1e3),
            ]
            return optim_cls(param_groups)

        return mk_optim

    return factory


def adam_factory(optim_cls):
    def factory(lr):
        def mk_optim(params):
            return optim_cls(params, lr=lr, weight_decay=lr / 1e3)

        return mk_optim

    return factory


OPTIMIZER_FACTORIES = {
    "adam": adam_factory(torch.optim.Adam),
    "adamw": adam_factory(torch.optim.AdamW),
    "muon": muon_param_group_factory(SingleDeviceMuonWithAuxAdam),
    "muonspectralnorm": muon_param_group_factory(
        SingleDeviceMuonWithAuxAdamSpectralNorm
    ),
    "muon10p": muon_param_group_factory(SingleDeviceMuonWithAuxAdam10p2),
    "10p": muon_param_group_factory(Adam10p),
    "spectralnorm": muon_param_group_factory(AdamWSpectralNorm),
}


def map_create_optimizer(optimizer_name):
    try:
        print(f"Using optimizer: {optimizer_name}")
        return OPTIMIZER_FACTORIES[optimizer_name]
    except KeyError:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
