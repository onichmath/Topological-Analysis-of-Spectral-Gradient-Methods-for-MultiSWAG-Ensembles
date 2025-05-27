from torch import nn


class MLP(nn.Module):
    def __init__(self, *args) -> None:
        super().__init__()
        kwargs = args[0]
        input_dim = kwargs.get("input_dim", 28 * 28)
        hidden_dim = kwargs.get("hidden_dim", 56)
        output_dim = kwargs.get("output_dim", 10)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.fc1 = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
        )
        # self.fc2 = nn.Sequential(
        #     nn.Linear(self.hidden_dim, self.hidden_dim),
        #     nn.ReLU(),
        # )
        self.fc3 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.fc2(x)
        x = self.fc3(x)
        return x
