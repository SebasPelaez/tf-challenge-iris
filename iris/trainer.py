import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from iris.data import LandMarkDataset
from iris.models.baseline import BaseLine
from iris.train_loop import TrainLoop


def main(
    kfold,
    batch_size,
    in_features,
    out_features,
    learning_rate,
    gamma,
    step_size,
    num_epochs,
    save_models,
):
    img_metadata = pd.read_csv("img_metadata.csv")

    # Creates dataset and dataloaders
    train_ds = LandMarkDataset(
        "dataset", img_metadata[img_metadata.iloc[:, 1] != kfold][:10]
    )
    test_ds = LandMarkDataset(
        "dataset", img_metadata[img_metadata.iloc[:, 1] == kfold][:10]
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # define model and move model to the right device
    model = BaseLine(
        use_pretrained=True, in_features=in_features, out_features=out_features
    ).to(device)

    # Define optmizer
    params = [params for params in model.parameters() if params.requires_grad]
    optimizer = Adam(params, **{"lr": learning_rate})
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, **{"gamma": gamma, "step_size": step_size}
    )

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

    main(
        kfold=0,
        batch_size=2,
        in_features=1000,
        out_features=20,
        learning_rate=0.001,
        gamma=0.1,
        step_size=100,
        num_epochs=3,
        save_models="saved_models",
    )
