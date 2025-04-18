from collections import OrderedDict
import torch

import flwr as fl
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from fl_draft.centralized import load_data, load_model, test, train


def get_parameters(model):
    model.train()
    return [val.cpu().numpy() for name, val in model.state_dict().items() if "bn" not in name]


def set_parameters(model, parameters):
    model.train()
    # keys = model.state_dict().keys()
    keys = [k for k in model.state_dict().keys() if "bn" not in k]
    params_dict = zip(keys, parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=False)

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):
    def __init__(self, model, trainloader, testloader, local_epochs):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        train_loss = train(self.model, self.trainloader, self.local_epochs, self.device)
        print(train_loss)
        if train_loss == None:
            train_loss = 0
        return get_parameters(self.model), len(self.trainloader.dataset), {"train_loss": float(train_loss)}

    def evaluate(self, parameters, config):
        set_parameters(self.model, parameters)
        loss, accuracy = test(self.model, self.testloader, self.device)
        return float(loss), len(self.testloader.dataset), {"accuracy": float(accuracy)}


def main() -> None:
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_model(DEVICE)
    trainloader, testloader = load_data(0)

    # Perform a single forward pass to properly initialize BatchNorm
    _ = model(next(iter(trainloader))["img"].to(DEVICE))

    # Start client
    client=FlowerClient(model, trainloader, testloader, local_epochs=3).to_client()
    fl.client.start_client(server_address="127.0.0.1:8080", client=client)


if __name__ == "__main__":
    main()