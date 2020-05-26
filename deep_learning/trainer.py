
import torch
import torch.nn.functional as F
from .metrics import Metrics, Accuracy, Precision, Recall, F1


class Trainer():
    def __init__(self, model, **model_kwargs):
        # Check CUDA availability and use it if possible
        cuda = True if torch.cuda.is_available() else False
        self.dev = torch.device("cpu") if not cuda else torch.device("cuda")
        print(f'Training on {self.dev} device')
        self.model = model.to(self.dev)

        self.board_idx = 0
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        self.epoch = 0
        self.train_metrics = {}
        self.val_metrics = {}

        self.loss_function = F.binary_cross_entropy_with_logits
        self.metrics = Metrics(Accuracy, Precision, Recall, F1)

    def train_epoch(self, train_loader):
        self.epoch += 1
        self.model.train(True)
        for iteration, (img, target) in enumerate(train_loader):
            img = img.to(self.dev, torch.float)
            target = target.to(self.dev, torch.float, non_blocking=True)

            self.opt.zero_grad()
            y_hat = self.model(img)
            loss = self.loss_function(y_hat, target)
            loss.backward()
            self.opt.step()

            self.metrics.step(y_hat, target, Loss=loss.detach())
            self.board_idx += img.shape[0]


    def val_epoch(self, val_loader):
        self.model.train(False)
        with torch.no_grad():
            for iteration, (img, target) in enumerate(val_loader):
                img = img.to(self.dev, torch.float)
                target = target.to(self.dev, torch.float, non_blocking=True)
                y_hat = self.model(img)

                loss = self.loss_function(y_hat, target)
                self.metrics.step(y_hat, target, Loss=loss.detach())
