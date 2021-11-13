from typing import Callable
from matplotlib.pyplot import axis
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module

from torch.optim.lr_scheduler import _LRScheduler
from pathlib import Path


class TrainLoop:
    def __init__(
        self,
        num_clases: int,
        train_dataloder: DataLoader,
        test_dataloder: DataLoader,
        loss_fun: Callable,
        device: str
    ) -> None:
        self.num_clases = num_clases
        self.train_dataloder = train_dataloder
        self.test_dataloder = test_dataloder

        self.train_dataloder_size = len(self.train_dataloder.dataset)
        self.test_dataloder_size = len(self.test_dataloder.dataset)

        self.device = device
        self.loss_fun = loss_fun

    def train_val(
        self,
        num_epochs: int,
        model: Module,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler,
        save_models: str,
    ):
        models_folder = Path(save_models)
        if not models_folder.exists():
            models_folder.mkdir()

        self.prev_loss = torch.tensor([float("inf")]).to(self.device)

        notrainable_total_params = sum(p.numel() for p in model.parameters())
        trainable_total_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )

        print(
            f"Number no trainable parameters: {notrainable_total_params} \nNumber trainable parameters: {trainable_total_params}"
        )

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")

            # train one epoch
            self._train_step(
                model,
                self.train_dataloder,
                self.loss_fun,
                self.device,
                optimizer,
                self.train_dataloder_size
            )

            # evaluate
            self.prev_loss = self._val_step(
                model,
                self.test_dataloder,
                self.loss_fun,
                self.device,
                models_folder,
                epoch,
                self.prev_loss,
                self.test_dataloder_size
            )
        
            # https://discuss.pytorch.org/t/how-to-use-torch-optim-lr-scheduler-exponentiallr/12444
            if lr_scheduler is not None:
                lr_scheduler.step()

        print("Done!")

    @staticmethod
    def _train_step(
        model: Module,
        data_loader: DataLoader,
        loss_fun: Callable,
        device: str,
        optimizer: Optimizer,
        dl_size
    ):

        running_loss, running_acc = 0, 0
        model.train()
        for batch, (X, y) in enumerate(data_loader):
            y = y.to(device)
            preds = model(X.float().to(device))
            loss = loss_fun(preds, y)

            _, y_pred = torch.max(preds, axis=1)

            # Sets gradients to zero
            optimizer.zero_grad()

            # backpropagation (computes derivates)
            loss.backward()

            # optimizer step (updates parameters)
            optimizer.step()

            if batch % 1 == 0:
                print(f"loss: {loss.item():>7f}  [{batch * len(X):>5d}/{dl_size:>5f}]")

            running_loss += loss.item() * X.size(0)
            running_acc += torch.sum(y_pred == y)
        
        epoch_loss = running_loss / dl_size
        epoch_acc = running_acc / dl_size
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            "Train", epoch_loss, epoch_acc))

    def val(
        self,
        model: Module,
        data_loader: DataLoader,
        loss_fun: Callable,
        device: str,
        models_folder: str,
        epoch: str,
        prev_loss: torch.tensor,
    ):
        self._val_step(
            model, data_loader, loss_fun, device, models_folder, epoch, prev_loss, self.test_dataloder_size
        )

    @staticmethod
    def _val_step(
        model: Module,
        data_loader: DataLoader,
        loss_fun: Callable,
        device: str,
        models_folder: str,
        epoch: str,
        prev_loss: torch.tensor,
        dl_size: int
    ):
        model.eval()
        
        running_loss, running_corrects = 0, 0
        with torch.no_grad():
            for batch, (X, y) in enumerate(data_loader):
                y = y.to(device)
                preds = model(X.float().to(device))
                _, y_preds = torch.max(preds, axis=1)
                
                running_loss += loss_fun(preds, y).item() * X.size(0) # computes sum of losses
                running_corrects += torch.sum(y_preds == y)

        epoch_loss = running_loss / dl_size # computes loss mean
        epoch_acc = running_corrects / dl_size # computes accuracy mean
        print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            "Validation", epoch_loss, epoch_acc))

        if epoch_loss < prev_loss:
            torch.save(
                model,
                f"{str(models_folder)}/{model.model_name}_epoch_{epoch}_metric_{epoch_acc}.pt",
            )

        return epoch_loss
