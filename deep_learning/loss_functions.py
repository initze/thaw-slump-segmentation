import torch.nn
import torch.nn.functional as F


def get_loss(loss_args):
    loss_type = loss_args['type']
    if loss_type in ('CrossEntropy') :
        loss_class = torch.nn.BCEWithLogitsLoss
        args = dict()
        if 'weights' in loss_args:
            args['weight'] = torch.tensor(loss_args['weights'])
    elif loss_type == 'AutoCE':
        return auto_ce
    else:
        print(f"No Loss of type {loss_type} known")

    return loss_class(**args)


def auto_ce(y_hat, y):
    with torch.no_grad():
        counts = torch.stack([(y == i).float().mean() for i in range(3)])
        weights = 1.0 / (3.0 * counts)
    return F.cross_entropy(y_hat, y.squeeze(1), weight=weights)
