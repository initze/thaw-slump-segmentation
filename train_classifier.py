import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from torchsummary import summary

from deep_learning.utils.data import PTDataset
from deep_learning.models import ResNet
import deep_learning.models.layers as layers

from tqdm.autonotebook import tqdm


def _update_dict(dictionary, key, value):
    if key in dictionary:
        dictionary[key].append(value)
    else:
        dictionary[key] = [value]


class TrainingState():
    def __init__(self, model_type, *model_args, **model_kwargs):
        cuda = True if torch.cuda.is_available() else False
        self.dev = torch.device("cpu") if not cuda else torch.device("cuda")
        print(f'Training on {self.dev} device')

        self.board_idx = 0
        self.model = model_type(*model_args, **model_kwargs).to(self.dev)
        self.opt = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        self.epoch = 0
        self.train_metrics = {}
        self.val_metrics = {}

        self.loss_function = F.binary_cross_entropy_with_logits

        self.batch_size = 4

    def eval_metrics(self, metrics_dict, prediction, target):
        with torch.no_grad():
            TP = ((prediction > 0) * (target > 0)).float().sum().item()
            FP = ((prediction > 0) * (target <= 0)).float().sum().item()
            FN = ((prediction <= 0) * (target > 0)).float().sum().item()
            TN = ((prediction <= 0) * (target <= 0)).float().sum().item()

            accuracy = (TP + TN) / (TP + FP + FN + TN)
            error = (FN + FP) / (TP + FP + FN + TN)
            precision = np.nan if TP + FP == 0 else TP / (TP + FP)
            recall = np.nan if TP + FN == 0 else TP / (TP + FN)
            f1 = np.nan if precision + recall == 0 else 2 * precision * recall / (precision + recall)

            _update_dict(metrics_dict, 'Accuracy', accuracy)
            _update_dict(metrics_dict, 'Error', error)
            _update_dict(metrics_dict, 'Precision', precision)
            _update_dict(metrics_dict, 'Recall', recall)
            _update_dict(metrics_dict, 'F1', f1)

    def train_epoch(self, train_loader):
        self.epoch += 1
        metrics = {}
        self.model.train(True)
        print(f'Starting train #{self.epoch}')
        for iteration, (img, target) in enumerate(train_loader):
            img    = img.to(self.dev, torch.float)
            target = target.max(dim=3)[0].max(dim=2)[0]
            target = target.to(self.dev, torch.float, non_blocking=True)

            self.opt.zero_grad()
            y_hat = self.model(img)
            bce_loss = self.loss_function(y_hat, target)

            loss = bce_loss
            loss.backward()
            self.opt.step()

            lossdict = {
                'BCE': bce_loss,
            }

            for key in lossdict:
                _update_dict(metrics, key, lossdict[key].item())
            self.eval_metrics(metrics, y_hat, target)

            self.board_idx += img.shape[0]
        for key in metrics:
            value = np.nanmean(metrics[key])
            _update_dict(self.train_metrics, key, value)

    def val_epoch(self, val_loader):
        metrics = {}
        self.model.train(False)
        print(f'Starting val #{self.epoch}')
        with torch.no_grad():
            for iteration, (img, target) in enumerate(val_loader):
                img    = img.to(self.dev, torch.float)
                target = target.max(dim=3)[0].max(dim=2)[0]
                target = target.to(self.dev, torch.float, non_blocking=True)

                y_hat = self.model(img)

                bce_loss = self.loss_function(y_hat, target)
                lossdict = {
                    'BCE': bce_loss,
                }
                for key in lossdict:
                    _update_dict(metrics, key, lossdict[key].item())
                self.eval_metrics(metrics, y_hat, target)

        for key in metrics:
            value = np.nanmean(metrics[key])
            _update_dict(self.val_metrics, key, value)


if __name__ == "__main__":
    model_type = ResNet
    state = TrainingState(model_type, 4, 1)
    summary(state.model, [(4, 256, 256)])

    train_loader = DataLoader(PTDataset('data/tiles_train', ['data', 'mask']),
                              batch_size=state.batch_size,
                              num_workers=4,
                              pin_memory=True)

    val_loader   = DataLoader(PTDataset('data/tiles_val', ['data', 'mask']),
                              batch_size=state.batch_size,
                              num_workers=4,
                              pin_memory=True)

    model_name = model_type.__name__
    for epoch in range(20):
        state.train_epoch(tqdm(train_loader))
        state.val_epoch(val_loader)
