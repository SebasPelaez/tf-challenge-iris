import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from iris.data import LandMarkDataset
from iris.models.baseline import BaseLine
from iris.train_loop import TrainLoop


def main(
    img_dir,
    img_metadata,
    batch_size,
    model_name,
    out_features,
    learning_rate,
    gamma,
    step_size,
    num_epochs,
    save_models,
):

    # Creates dataset and dataloaders
    train_ds = LandMarkDataset(
        img_dir, img_metadata[0]
    )
    test_ds = LandMarkDataset(
        img_dir, img_metadata[1]
    )
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # define model and move model to the right device
    model = BaseLine(
        model_name=model_name, use_pretrained=True, out_features=out_features
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

    img_metadata = pd.read_csv("img_metadata_train_dev.csv")
    train_img_metadata = img_metadata[img_metadata.iloc[:, 1] != 0][:100]
    test_img_metadata = img_metadata[img_metadata.iloc[:, 1] == 0][:100]

    main(
        img_dir="dataset/train/",
        img_metadata=(train_img_metadata, test_img_metadata),
        batch_size=12,
        model_name="resnet18",
        out_features=21,
        learning_rate=0.001,
        gamma=0.1,
        step_size=100,
        num_epochs=3,
        save_models="saved_models",
    )
