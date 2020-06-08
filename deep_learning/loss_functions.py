import torch.nn


def get_loss(loss_args):
    loss_type = loss_args['type']
    if loss_type == 'BinaryCrossEntropy':
        loss_class = torch.nn.BCEWithLogitsLoss
        args = dict()
        if 'pos_weight' in loss_args:
            args['pos_weight'] = loss_args['pos_weight'] * torch.ones([])
    else:
        print(f"No Loss of type {loss_type} known")

    return loss_class(**args)
