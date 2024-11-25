import torch
import torch.nn.functional as F
from torch import nn

from LT.models import resnet32
from LT.models.reactnet_optimized import Reactnet, binaryconv3x3, LearnableBias, BasicBlock


class BYOLTrainer(nn.Module):
    def __init__(self, online_network, target_network, device, momentum=0.996, mlp_hidden_size=512, use_sup_logit=False, num_classes=10):
        super(BYOLTrainer, self).__init__()
        self.mlp_hidden_size = mlp_hidden_size
        self.online_network = self._add_projection_layer(online_network)
        self.target_network = self._add_projection_layer(target_network)
        self.use_sup_logit = use_sup_logit
        ################
        # sanity check #
        ################
        tmp_inp = torch.randn(2, 3, 32, 32)
        check = CheckOutput(list(online_network.children())[-2])
        online_network.eval()
        self.online_network.eval()
        out1 = self.online_network.encoder(tmp_inp)
        online_network(tmp_inp)
        out2 = check.output
        assert torch.equal(out1, out2)

        self.device = device
        self.predictor = MLPHead(in_channels=online_network.fc.in_features, mlp_hidden_size=self.mlp_hidden_size,
                                 projection_size=online_network.fc.in_features).to(device)
        self.m = momentum
        self._initializes_target_network()
        if self.use_sup_logit:
            self.classifier = nn.Linear(online_network.fc.in_features, num_classes)

        ########################
        # remove unnecessaries #
        ########################
        check.remove()
        del check
        del online_network
        del target_network

    def _add_projection_layer(self, model):
        tmp_model = nn.Module()
        if model.__class__.__name__.lower() == 'reactnet':
            tmp_model.encoder = torch.nn.Sequential(*list(model.feature.children()), model.pool1)
        else:
            tmp_model.encoder = torch.nn.Sequential(*list(model.children())[:-1])
        tmp_model.projection = MLPHead(in_channels=model.fc.in_features, mlp_hidden_size=self.mlp_hidden_size,
                                       projection_size=model.fc.in_features)

        def forward(x):
            x = tmp_model.encoder(x)
            x = tmp_model.projection(x)
            return x

        tmp_model.forward = forward
        return tmp_model

    @torch.no_grad()
    def _update_target_network_parameters(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @staticmethod
    def regression_loss(x, y):
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)
        return 2 - 2 * (x * y).sum(dim=-1)

    def _initializes_target_network(self):
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

    def forward(self, batch_view_1, batch_view_2):

        batch_view_1 = batch_view_1.to(self.device)
        batch_view_2 = batch_view_2.to(self.device)
        if self.use_sup_logit:
            loss, sup_logit = self.update(batch_view_1, batch_view_2)
            return loss, sup_logit
        else:
            loss = self.update(batch_view_1, batch_view_2)
            return loss

    def update(self, batch_view_1, batch_view_2):
        # compute query feature
        predictions_from_view_1 = self.predictor(self.online_network(batch_view_1))
        predictions_from_view_2 = self.predictor(self.online_network(batch_view_2))
        if self.use_sup_logit:
            sup_logit = self.online_network.encoder(batch_view_1)
        # compute key features
        with torch.no_grad():
            targets_to_view_2 = self.target_network(batch_view_1)
            targets_to_view_1 = self.target_network(batch_view_2)

        loss = self.regression_loss(predictions_from_view_1, targets_to_view_1)
        loss += self.regression_loss(predictions_from_view_2, targets_to_view_2)
        if self.use_sup_logit:
            return loss.mean(), sup_logit
        else:
            return loss.mean()

    def parameters(self):
        return list(self.online_network.parameters()) + list(self.predictor.parameters()) + list(self.classifier.parameters())


class MLPHead(nn.Module):
    def __init__(self, in_channels, mlp_hidden_size, projection_size):
        super(MLPHead, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(in_channels, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )

    def forward(self, x):
        x = x.squeeze()
        return self.net(x)


class CheckOutput:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook)

    def hook(self, module, input, output):
        self.output = output

    def remove(self):
        self.hook.remove()


if __name__ == '__main__':
    import torchvision.models as models

    device = 'cpu'

    online_model = Reactnet()
    target_model = Reactnet()

    byol = BYOLTrainer(online_network=online_model, target_network=target_model, device=device)

    optim = torch.optim.SGD(byol.parameters(), lr=0.5, momentum=0.9)

    inp1 = torch.randn(2, 3, 32, 32)
    inp2 = torch.randn(2, 3, 32, 32)

    loss = byol(inp1, inp2)
    optim.zero_grad()
    loss.backward()
    # print(list(byol.predictor.parameters())[0])
    # print(list(byol.online_network.parameters())[0][0])
    ##########################################
    # two lines must be called in same order #
    ##########################################
    optim.step()
    # print(list(byol.predictor.parameters())[0])
    # print(list(byol.online_network.parameters())[0][0])

    # print(list(byol.target_network.parameters())[0][0])
    byol._update_target_network_parameters()
    # print(list(byol.target_network.parameters())[0][0])

    st = byol.state_dict()
    byol.load_state_dict(st)