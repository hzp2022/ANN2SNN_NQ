from torch import nn
import torch.nn.functional as F
import torch
from spikingjelly.clock_driven import neuron
from torch.autograd import Function


class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input


class IFNeuron(nn.Module):
    def __init__(self, scale=1.):
        super(IFNeuron, self).__init__()
        self.up = scale
        self.t = 0
        self.neuron = neuron.IFNode(v_reset=None)

    def forward(self, x):
        x = x / self.up
        if self.t == 0:
            self.neuron(torch.ones_like(x) * 0.5)

        self.t += 1

        x = self.neuron(x)

        return x * self.up

    def reset(self):
        self.t = 0
        self.neuron.reset()


class FloorLayer(Function):
    @staticmethod
    def forward(ctx, input):
        return input.floor()

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


qcfs = FloorLayer.apply


class QCFS(nn.Module):
    def __init__(self, up=8., t=32, sigma=0.01):
        super().__init__()
        self.up = nn.Parameter(torch.tensor([up]), requires_grad=True)
        self.t = t
        self.sigma = sigma

    def forward(self, x):
        x = x / self.up
        x = torch.clamp(x, 0, 1)
        x = qcfs(x * self.t + 0.5) / self.t
        x = x * self.up
        noise = torch.normal(mean=0., std=1.0, size=x.shape, device='cuda:0')
        x = x + noise * self.sigma
        return x


class MPLayer(nn.Module):
    def __init__(self, val_sim_len, batchsize, up=8., L=32, sigma=0., first_activation_layer=False):
        super().__init__()
        self.first_activation_layer = first_activation_layer
        self.up = nn.Parameter(torch.tensor([up]), requires_grad=True)
        self.L = L
        self.sigma = sigma

        self.t = 0
        self.neuron = neuron.IFNode(v_reset=None)
        self.val_sim_len = val_sim_len
        self.batchsize = batchsize

        self.snn_rate = None
        self.snn_mode = False
        self.gaps_var = []
        self.train_mode = True
        self.gaps_mean = []

        self.mu = 0.
        self.pre2_mu = 0.
        self.pre2_sigma = 0.

    def forward(self, x):
        if self.snn_mode:
            with torch.no_grad():
                output_seq = []
                x = x / self.up

                if self.t == 0:
                    if x.dim() == 4:
                        if x.shape[0] <= self.batchsize and self.first_activation_layer:
                            x = x.unsqueeze(0).repeat(self.val_sim_len, 1, 1, 1, 1)
                        else:
                            x = x.reshape(self.val_sim_len, x.shape[0] // self.val_sim_len, x.shape[1], x.shape[2], x.shape[3])
                    else:
                        x = x.reshape(self.val_sim_len, x.shape[0] // self.val_sim_len, x.shape[1])

                    self.neuron.reset()
                    self.neuron(torch.ones_like(x[0]) * 0.5)

                self.snn_rate = torch.zeros_like(x[0])

                for t in range(self.val_sim_len):
                    output = self.neuron(x[t])
                    output_seq.append(output)
                    self.t += 1

                    if t < self.val_sim_len:
                        self.snn_rate += output

                self.snn_rate = self.snn_rate * self.up / self.val_sim_len

                if self.t == self.val_sim_len:
                    self.t = 0

                output_seq = torch.stack(output_seq)

                if output_seq.dim() == 5:
                    output_seq = output_seq.reshape(output_seq.shape[0] * output_seq.shape[1], output_seq.shape[2],
                                                    output_seq.shape[3], output_seq.shape[4])
                else:
                    output_seq = output_seq.reshape(output_seq.shape[0] * output_seq.shape[1], output_seq.shape[2])

                return output_seq * self.up
        else:
            if self.train_mode:
                x = x / self.up
                x = torch.clamp(x, 0, 1)
                x = qcfs(x * self.L + 0.5) / self.L
                x = x * self.up
                noise = torch.normal(mean=0., std=1.0, size=x.shape, device='cuda:0')
                x = x - noise * self.sigma
                return x
            else:
                with torch.no_grad():
                    x = x / self.up
                    x = torch.clamp(x, 0, 1)
                    x = qcfs(x * self.L + 0.5) / self.L
                    x = x * self.up

                    gap = x - self.snn_rate

                    var_deviation = torch.var(gap)
                    self.gaps_var.append(var_deviation)

                    mean = torch.mean(gap)
                    self.gaps_mean.append(mean)

                    return x


