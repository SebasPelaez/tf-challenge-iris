from typing import Callable
import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.nn import Module

from torch.optim.lr_scheduler import _LRScheduler


class TrainLoop:
    def __init__(
        self,
        num_clases: int,
        train_dataloder: DataLoader,
        test_dataloder: DataLoader,
        loss_fun: Callable,
    ) -> None:
        self.num_clases = num_clases
        self.train_datagen = train_dataloder
        self.test_datagen = test_dataloder
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loss_fun = loss_fun

    def train_val(
        self,
        num_epochs: int,
        model: Module,
        optimizer: Optimizer,
        lr_scheduler: _LRScheduler,
        device: str,
    ):

        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}\n-------------------------------")

            # train one epoch
            self._train_step(
                model,
                self.train_datagen,
                self.loss_fun,
                device,
                optimizer,
                lr_scheduler,
            )

            # evaluate
            self._val_step(model, self.test_datagen, self.loss_fun, device)

        print("Done!")

    @staticmethod
    def _train_step(
        model: Module,
        data_loader: DataLoader,
        loss_fun: Callable,
        device: str,
        optimizer: Optimizer,
        lr_scheduler=None,
    ):
        model.train()
        size = len(data_loader.dataset)
        for batch, (X, y) in enumerate(data_loader):
            pred = model(X.float().to(device))
            loss = loss_fun(pred, y.to(device))

            # Sets gradients to zero
            model.zero_grad()

            # backpropagation (computes derivates)
            loss.backward()

            # optimizer step (updates parameters)
            optimizer.step()

            if lr_scheduler is not None:
                lr_scheduler.step()

            if batch % 1 == 0:
                loss, current = loss.item(), batch * len(X)
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5f}]")

    def val(self, model: Module, data_loader: DataLoader, loss_fun: Callable):
        self._val_step(model, data_loader, loss_fun)

    @staticmethod
    def _val_step(
        model: Module, data_loader: DataLoader, loss_fun: Callable, device: str
    ):
        model.eval()
        size = len(data_loader.dataset)
        test_loss, correct = 0, 0
        with torch.no_grad():
            for batch, (X, y) in enumerate(data_loader):
                pred = model(X.float().to(device))
                y = y.to(device)
                test_loss += loss_fun(pred, y)
                correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        test_loss = test_loss / batch
        correct /= size
        print(
            f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
        )