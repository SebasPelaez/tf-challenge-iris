from torch.nn import Module, Linear
from torchvision import models


class BaseLine(Module):
    def __init__(self, model_name, use_pretrained, out_features=2) -> None:
        super(BaseLine, self).__init__()

        self.model_name = model_name
        model = getattr(models, model_name)
        self.model = model(pretrained=use_pretrained)
        last_layer_name, last_layer = list(self.model.named_modules())[-1]

        if use_pretrained:
            for param in self.model.parameters():
                param.requires_grad = False

        setattr(
            self.model,
            last_layer_name,
            Linear(
                in_features=last_layer.in_features, out_features=out_features, bias=True
            ),
        )

    def forward(self, x):
        # takes input with shape (3, ?, ?)
        preds = self.model(x)
        return preds
