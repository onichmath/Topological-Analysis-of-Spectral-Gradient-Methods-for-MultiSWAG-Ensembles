import torch
from muon import UnifiedOptimizer


def split_params(params):
    param_list = list(params)
    hidden_weights = [p for p in param_list if p.ndim >= 2]
    others = [p for p in param_list if p.ndim < 2]
    return hidden_weights, others


def unified_optimizer_factory(method="ns", top_percent=1.0):
    """Factory for UnifiedOptimizer with NS or NS_FAST method."""

    def factory(lr):
        def mk_optim(params):
            hidden_weights, others = split_params(params)
            param_groups = []

            if hidden_weights:
                param_groups.append(
                    dict(
                        params=hidden_weights,
                        use_muon=True,
                        lr=lr,
                        weight_decay=lr / 1e3,
                        momentum=0.95,
                        ns_steps=10,
                        method=method,
                        top_percent=top_percent,
                    )
                )

            if others:
                param_groups.append(
                    dict(
                        params=others,
                        use_muon=False,
                        lr=lr,
                        weight_decay=lr / 1e3,
                        betas=(0.9, 0.95),
                        eps=1e-10,
                    )
                )

            return UnifiedOptimizer(param_groups)

        return mk_optim

    return factory


def adam_factory(optim_cls):
    def factory(lr):
        def mk_optim(params):
            return optim_cls(params, lr=lr, weight_decay=lr / 1e3)

        return mk_optim

    return factory


def adam_svd_factory(top_percent=0.1):
    """Factory for AdamW optimizer with SVD filtering post-processing."""

    def factory(lr):
        def mk_optim(params):
            from muon import orthogonalize

            class AdamWSVD(torch.optim.AdamW):
                def __init__(self, params, lr, top_percent, **kwargs):
                    super().__init__(params, lr=lr, **kwargs)
                    self.top_percent = top_percent

                @torch.no_grad()
                def step(self, closure=None):
                    loss = super().step(closure)

                    for group in self.param_groups:
                        for p in group["params"]:
                            if p.grad is not None and p.grad.ndim >= 2:
                                p.grad.data = orthogonalize(
                                    p.grad.data,
                                    method="svd",
                                    top_percent=self.top_percent,
                                )

                    return loss

            return AdamWSVD(
                params, lr=lr, top_percent=top_percent, weight_decay=lr / 1e3
            )

        return mk_optim

    return factory


def adam_svd_factory_fast(top_percent=0.1):
    """Factory for AdamW optimizer with fast approximate SVD filtering."""

    def factory(lr):
        def mk_optim(params):
            from muon import orthogonalize

            class AdamWSVDFast(torch.optim.AdamW):
                def __init__(self, params, lr, top_percent, **kwargs):
                    super().__init__(params, lr=lr, **kwargs)
                    self.top_percent = top_percent

                @torch.no_grad()
                def step(self, closure=None):
                    loss = super().step(closure)

                    for group in self.param_groups:
                        for p in group["params"]:
                            if p.grad is not None and p.grad.ndim >= 2:
                                p.grad.data = orthogonalize(
                                    p.grad.data,
                                    method="svd_fast",
                                    top_percent=self.top_percent,
                                )

                    return loss

            return AdamWSVDFast(
                params, lr=lr, top_percent=top_percent, weight_decay=lr / 1e3
            )

        return mk_optim

    return factory


OPTIMIZER_FACTORIES = {
    "adam": adam_factory(torch.optim.Adam),
    "adamw": adam_factory(torch.optim.AdamW),
    "muon": unified_optimizer_factory(method="ns", top_percent=1.0),
    "muon10p": unified_optimizer_factory(method="ns_fast", top_percent=0.1),
    "muonspectralnorm": unified_optimizer_factory(method="ns_fast", top_percent=0.0),
    "10p": adam_svd_factory_fast(top_percent=0.1),
    "spectralnorm": adam_svd_factory_fast(top_percent=0.0),
}


def map_create_optimizer(optimizer_name):
    try:
        print(f"Using optimizer: {optimizer_name}")
        return OPTIMIZER_FACTORIES[optimizer_name]
    except KeyError:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
