import torch
import torch.nn as nn
import torch.nn.functional as F


class MarchingSquares(nn.Module):
    def __init__(self, input_shape):
        super().__init__()
        self.register_buffer('cell_mask', torch.Tensor([1, 2, 4, 8]).to(torch.short))

        yticks = torch.arange(-0.5, input_shape[0])
        xticks = torch.arange(-0.5, input_shape[1])
        grid = torch.stack(torch.meshgrid(yticks, xticks), dim=-1)
        self.register_buffer('grid', grid)

        SW, SE, NE, NW = range(4)
        self.segments = [[] for _ in range(16)]
        self.segments[0b0000] = []
        self.segments[0b0001] = [[[NW, SW], [SE, SW]]]
        self.segments[0b0010] = [[[SW, SE], [NE, SE]]]
        self.segments[0b0011] = [[[NW, SW], [NE, SE]]]
        self.segments[0b0100] = [[[SE, NE], [NW, NE]]]
        self.segments[0b0101] = [[[NW, SW], [SE, SW]], [[SE, NE], [NW, NE]]]
        self.segments[0b0110] = [[[SW, SE], [NW, NE]]]
        self.segments[0b0111] = [[[NW, SW], [NW, NE]]]
        self.segments[0b1000] = [[[NE, NW], [SW, NW]]]
        self.segments[0b1001] = [[[NE, NW], [SE, SW]]]
        self.segments[0b1010] = [[[SW, SE], [NE, SE]], [[NE, NW], [SW, NW]]]
        self.segments[0b1011] = [[[NE, NW], [NE, SE]]]
        self.segments[0b1100] = [[[SE, NE], [SW, NW]]]
        self.segments[0b1101] = [[[SE, NE], [SE, SW]]]
        self.segments[0b1110] = [[[SW, SE], [SW, NW]]]
        self.segments[0b1111] = []

    def forward(self, coarse):
        coarse = F.pad(coarse, [1, 1, 1, 1], mode='replicate')
        nw = coarse[:, :, :-1, :-1]
        ne = coarse[:, :, :-1, 1:]
        sw = coarse[:, :, 1:, :-1]
        se = coarse[:, :, 1:, 1:]
        values = torch.stack([sw, se, ne, nw], dim=-1)

        cells = (values > 0).to(torch.short)
        cellidx = torch.tensordot(cells, self.cell_mask, dims=([-1], [0]))

        batch_idx = []
        edges = []
        for b in range(cells.shape[0]):
            for y in range(cells.shape[2]):
                for x in range(cells.shape[3]):
                    idx = cellidx[b, 0, y, x]
                    segments = self.segments[idx]
                    if len(segments) == 0:
                        continue
                    vals = values[b, 0, y, x]
                    points = torch.tensor([[1, 0], [1, 1], [0, 1], [0, 0]]) + self.grid[y, x]
                    for (l1, u1), (l2, u2) in segments:
                        start = (vals[u1] * points[u1] - vals[l1] * points[l1]) / (vals[u1] - vals[l1])
                        stop  = (vals[u2] * points[u2] - vals[l2] * points[l2]) / (vals[u2] - vals[l2])
                        batch_idx.append(torch.Tensor([b]).to(torch.short))
                        edge = torch.cat([start, stop], dim=0)
                        edges.append(edge)
        if len(batch_idx) == 0:
            return None
        else:
            return torch.stack(batch_idx, dim=0), torch.stack(edges, dim=0)
