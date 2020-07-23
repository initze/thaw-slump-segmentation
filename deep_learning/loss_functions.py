import torch.nn


def get_loss(loss_args):
    loss_type = loss_args['type']
    if loss_type in ('BinaryCrossEntropy', 'BCE') :
        loss_class = torch.nn.BCEWithLogitsLoss
        args = dict()
        if 'pos_weight' in loss_args:
            args['pos_weight'] = loss_args['pos_weight'] * torch.ones([])
    elif loss_type == 'AutoBCE':
        return auto_weight_bce
    else:
        print(f"No Loss of type {loss_type} known")

    return loss_class(**args)


def auto_weight_bce(y_hat_log, y):
    with torch.no_grad():
        beta = y.mean(dim=[2, 3], keepdims=True)
    logit_1 = F.logsigmoid(y_hat_log)
    logit_0 = F.logsigmoid(-y_hat_log)
    loss = -(1 - beta) * logit_1 * y \
           - beta * logit_0 * (1 - y)
    return loss.mean()

