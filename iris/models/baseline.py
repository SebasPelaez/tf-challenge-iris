from torch.nn import Module, Linear, Sequential
from torchvision import models


class BaseLine(Module):
    def __init__(self, use_pretrained, in_features=4096, out_features=2) -> None:
        super(BaseLine, self).__init__()
        model = models.resnet18(pretrained=use_pretrained)
        if use_pretrained:
            for param in model.parameters():
                param.requires_grad = False

        self.model = Sequential(model, Linear(in_features, out_features))

    def forward(self, x):
        # takes input with shape (3, ?, ?)
        preds = self.model(x)
        return preds
