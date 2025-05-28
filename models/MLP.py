from torch import nn


class MLP(nn.Module):
    def __init__(self, *args) -> None:
        super().__init__()

        kwargs = args[0]
        input_dim = kwargs.get("input_dim")
        hidden_dim = kwargs.get("hidden_dim")
        output_dim = kwargs.get("output_dim")
        num_hidden_layers = kwargs.get("num_hidden_layers")

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        layers = []
        dim = input_dim

        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(dim, self.hidden_dim))
            layers.append(nn.ReLU())
            dim = self.hidden_dim

        layers.append(nn.Linear(self.hidden_dim, self.output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
