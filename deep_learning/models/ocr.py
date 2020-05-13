import torch
import torch.nn as nn
import torch.nn.functional as F


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class SpatialSoftmax(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.softmax(x.view(*x.shape[:2], -1), dim=-1).view_as(x)


class _cbr1(nn.Module):
    def __init__(self, c_in, c_out, bn=True):
        super().__init__()
        self.bn = nn.BatchNorm1d(c_out) if bn else Identity()
        bias = not bn
        self.conv = nn.Conv1d(c_in, c_out, 1, bias=bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x


class _cbr2(nn.Module):
    def __init__(self, c_in, c_out, bn=True):
        super().__init__()
        self.bn = nn.BatchNorm2d(c_out) if bn else Identity()
        bias = not bn
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1, bias=bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x


class _down(nn.Module):
    def __init__(self, c_in, c_out, bn=True):
        super().__init__()
        self.bn = nn.BatchNorm2d(c_out) if bn else Identity()
        bias = not bn
        self.conv = nn.Conv2d(c_in, c_out, 2, stride=2, bias=bias)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        return x


class _Backbone(nn.Module):
    def __init__(self, scl):
        super().__init__()
        self.pre = nn.Sequential(
            _cbr2(2,       1 * scl),
            _down(1 * scl, 2 * scl),
            _cbr2(2 * scl, 2 * scl),
        )

        self.pyramid = nn.ModuleList([
            nn.Sequential(_cbr2(2 * scl, 2 * scl), _down( 2 * scl, 2 * scl)),
            nn.Sequential(_cbr2(2 * scl, 2 * scl), _down( 2 * scl, 4 * scl)),
            nn.Sequential(_cbr2(4 * scl, 4 * scl), _down( 4 * scl, 4 * scl)),
        ])

        self.skip_pre = _cbr2(2 * scl, 4 * scl)

        self.skip = nn.ModuleList([
            _cbr2( 2 * scl, 4 * scl),
            _cbr2( 4 * scl, 4 * scl),
            Identity()
        ])

    def forward(self, x):
        x = self.pre(x)
        scale = 2
        features = [F.interpolate(self.skip_pre(x), scale_factor=scale, mode='bilinear')]
        for pyramid, skip in zip(self.pyramid, self.skip):
            x = pyramid(x)
            scale *= 2
            features.append(F.interpolate(skip(x), scale_factor=scale, mode='bilinear'))
        return torch.cat(features, dim=1)


class OCRNet(nn.Module):
    def __init__(self, scl=16):
        super().__init__()

        K = 2
        S = 4 * scl
        Q = 4 * scl

        self.backbone = _Backbone(scl)
        self.soft_object_regions = nn.Sequential(
            _cbr1(16 * scl, 4 * scl),
            nn.Conv1d(4 * scl, K, 1),
        )

        self.pixel_representations = _cbr1(16 * scl, S)

        self.phi = _cbr1(S, Q)
        self.psi = _cbr1(S, Q)

        self.delta = _cbr1(S, Q)
        self.rho = _cbr1(Q, S)

        self.g = _cbr1(S + S, S)

        self.classify = nn.Conv2d(S, 2, 1)

    def forward(self, x):
        features = self.backbone(x)
        # Fold spatial dimension into one
        features = features.view(*features.shape[:2], -1)

        # sor: N x K x WH
        sor_logit = self.soft_object_regions(features)
        sor = torch.softmax(sor_logit, dim=2)
        # pr: N x S x WH
        pr = self.pixel_representations(features)

        # orr: N x K x S
        orr = torch.matmul(pr, sor.transpose(1, 2))

        query = self.phi(pr)
        key = self.psi(orr)

        # Pixel - Region Relation
        kappa = torch.matmul(key.transpose(1, 2), query)
        w = torch.softmax(kappa, dim=2)

        delta = self.delta(orr)
        pre_rho = torch.matmul(delta, w)
        y = self.rho(pre_rho)

        fused = torch.cat([pr, y], dim=1)
        g = self.g(fused).view(x.shape[0], -1, *x.shape[2:])
        final = torch.log_softmax(self.classify(g), dim=1)

        return final, torch.log_softmax(sor_logit.view_as(final), dim=1)

    def loss_eval(self, x, target, loss_fn=F.nll_loss):
        prediction, sor = self.forward(x)
        bce = loss_fn(prediction, target)
        soft_bce = loss_fn(sor, target)
        return prediction[:, 1] - prediction[:, 0], bce + 0.2 * soft_bce, dict(BCE=bce, SoftBCE=soft_bce)
