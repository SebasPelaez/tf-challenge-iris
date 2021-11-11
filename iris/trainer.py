import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.nn import Sequential, Dropout, Linear
from torch.utils.data import sampler
from torch.utils.data.sampler import WeightedRandomSampler

from iris.data import LandMarkDataset
from iris.models.baseline import BaseLine
from iris.train_loop import TrainLoop

import torchvision.transforms as transforms


def main(
    img_dir: str,
    img_metadata: pd.DataFrame,
    train_trans: transforms.Compose,
    dev_trans: transforms.Compose,
    batch_size: int,
    model: torch.nn.Module,
    out_features: int,
    optimizer_params: dict,
    lr_scheduler_params: dict,
    num_epochs: int,
    save_models: str,
    features_weights: list = None,
):
    sampler = (
        WeightedRandomSampler(features_weights, img_metadata[0].shape[0])
        if features_weights is not None
        else None
    )

    # Creates dataset and dataloaders
    train_ds = LandMarkDataset(img_dir, img_metadata[0], train_trans)
    test_ds = LandMarkDataset(img_dir, img_metadata[1], dev_trans)
    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Define optmizer
    params = [params for params in model.parameters() if params.requires_grad]
    optimizer = SGD(params, **optimizer_params)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **lr_scheduler_params)

    # Train-val step
    train_loop = TrainLoop(
        num_clases=out_features,
        train_dataloder=train_dl,
        test_dataloder=test_dl,
        loss_fun=F.cross_entropy,
    )
    train_loop.train_val(
        num_epochs=num_epochs,
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        device=device,
        save_models=save_models,
    )


if __name__ == "__main__":

    img_metadata = pd.read_csv("img_metadata_train_dev.csv")
    train_img_metadata = img_metadata[img_metadata.iloc[:, 1] == 0][:100]
    test_img_metadata = img_metadata[img_metadata.iloc[:, 1] == 0][:100]
    features_weights = img_metadata[img_metadata.iloc[:, 1] == 0].iloc[:100, 4]

    train_trans = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    # define model and move model to the right device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_features = 21
    model = BaseLine(
        model_name="densenet121",
        use_pretrained=True,
        last_layer={"classifier": Linear(in_features=1024, out_features=out_features)},
    ).to(device)

    main(
        img_dir="dataset/train/",
        img_metadata=(train_img_metadata, test_img_metadata),
        train_trans=train_trans,
        dev_trans=train_trans,
        batch_size=12,
        model=model,
        out_features=out_features,
        optimizer_params={"lr": 0.001, "momentum": 0.9},
        lr_scheduler_params={"gamma": 0.1, "step_size": 500, "verbose": True},
        num_epochs=10,
        save_models="saved_models",
        features_weights=features_weights,
    )
