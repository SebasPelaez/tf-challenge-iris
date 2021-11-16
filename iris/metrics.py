import torch 
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import scikitplot as skplt
import matplotlib.pyplot as plt
import seaborn as sn


from iris.data import LandMarkDataset, labels

def main(img_metadata, model_name, fig_name):
    # define model and move model to the right device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dev_trans =  A.Compose(
        [
            A.Resize(224, 224),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ]
    )

    model = torch.load(model_name)


    test_ds = LandMarkDataset("data/train/", img_metadata, dev_trans)
    test_dl = DataLoader(test_ds, batch_size=8, shuffle=False)


    preds_list, labels_list = [], []
    for idx, data in enumerate(test_dl):
        X, y = data
        preds = model(X.to(device))
        preds_list.extend(torch.argmax(preds, axis=1).tolist())
        labels_list.extend(y.tolist())


    cm = confusion_matrix(labels_list, preds_list)
    plt.figure(figsize=(20, 15))
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(pd.DataFrame(cm), annot=True, annot_kws={"size": 15}) # font size
    plt.savefig(fig_name)


if __name__ == "__main__":

    img_metadata = pd.read_csv("img_metadata_train_dev.csv")
    train_img_metadata = img_metadata[img_metadata.iloc[:, 1] != 0]
    test_img_metadata = img_metadata[img_metadata.iloc[:, 1] == 0]

    model_name_ext = "torchvision.models.densenet_epoch_24_metric_0.6017699241638184"
    main(
        img_metadata = train_img_metadata, 
        model_name=f"saved_models/{model_name_ext}.pt",
        fig_name=f"img/train_{model_name_ext}.png")
    
    main(
        img_metadata = test_img_metadata, 
        model_name=f"saved_models/{model_name_ext}.pt",
        fig_name=f"img/test_{model_name_ext}.png")