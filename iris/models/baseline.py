from torch.nn import Module, Linear
from torchvision import models


class BaseLine(Module):
    def __init__(self, model_name, use_pretrained, last_layer) -> None:
        super(BaseLine, self).__init__()

        self.model_name = model_name
        model = getattr(models, model_name)
        self.model = model(pretrained=use_pretrained)

        if use_pretrained:
            for param in self.model.parameters():
                param.requires_grad = False

        for last_layer_name, layer in last_layer.items():
            setattr(
                self.model,
                last_layer_name,
                layer,
            )

    def forward(self, x):
        # takes input with shape (3, ?, ?)
        preds = self.model(x)
        return preds
