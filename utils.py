import torch
import torch.nn as nn
from modules import *
import numpy as np
import math


def isActivation(name):
    if 'relu' in name.lower() or 'qcfs' in name.lower() or 'qcfs_with_noise' in name.lower() or 'mplayer' in name.lower():
        return True
    return False


def replace_MPLayer_by_neuron(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_MPLayer_by_neuron(module)
        if module.__class__.__name__ == 'MPLayer':
            model._modules[name] = IFNeuron(scale=module.v_threshold)
    return model


def replace_activation_by_neuron(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_activation_by_neuron(module)
        if isActivation(module.__class__.__name__.lower()):
            model._modules[name] = IFNeuron(scale=module.up.item())
    return model


def replace_activation_by_MPLayer(model, val_sim_len, batchsize, L, first=True):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name], first = replace_activation_by_MPLayer(module, val_sim_len, batchsize, L, first)
        if isActivation(module.__class__.__name__.lower()):
            if first:
                model._modules[name] = MPLayer(val_sim_len, batchsize, up=8., L=L, first_activation_layer=True)
                first = False
            else:
                model._modules[name] = MPLayer(val_sim_len, batchsize, up=8., L=L, first_activation_layer=False)

    return model, first


def replace_maxpool2d_by_avgpool2d(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            model._modules[name] = replace_maxpool2d_by_avgpool2d(module)
        if module.__class__.__name__ == 'MaxPool2d':
            model._modules[name] = nn.AvgPool2d(kernel_size=module.kernel_size,
                                                stride=module.stride,
                                                padding=module.padding)
    return model


def replace_activation_by_floor(model, t, std_deviations, up):
    for name, module in model._modules.items():
        i = 0
        if hasattr(module, "_modules"):
            model._modules[name] = replace_activation_by_floor(module, t, std_deviations, up)
        if isActivation(module.__class__.__name__.lower()):
            model._modules[name] = QCFS(up=up[i], t=t, sigma=std_deviations[i]).cuda()
            i += 1
    return model


def reset_net(model):
    for name, module in model._modules.items():
        if hasattr(module, "_modules"):
            reset_net(module)
        if 'Neuron' in module.__class__.__name__:
            module.reset()
    return model


def reset_MPLayer_neuron(model):
    with torch.no_grad():
        for name, module in model._modules.items():
            if hasattr(module, "_modules"):
                reset_MPLayer_neuron(module)
            if module.__class__.__name__ == 'MPLayer':
                module.neuron.reset()
    return model


def error(info):
    print(info)
    exit(1)


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def calculate_MPLayer(model, std_deviations):
    with torch.no_grad():
        for name, module in model._modules.items():
            if hasattr(module, "_modules"):
                calculate_MPLayer(module, std_deviations)
            if module.__class__.__name__ == 'MPLayer':
                std_deviation = 0.
                std_deviations.append(std_deviation)

    return std_deviations


def reset_gaps_std_and_mean(model):
    with torch.no_grad():
        for name, module in model._modules.items():
            if hasattr(module, "_modules"):
                reset_gaps_std_and_mean(module)
            if module.__class__.__name__ == 'MPLayer':
                module.gaps_std = []
                module.gaps_mean = []

    return model


def reset_gaps_var_and_mean(model):
    with torch.no_grad():
        for name, module in model._modules.items():
            if hasattr(module, "_modules"):
                reset_gaps_var_and_mean(module)
            if module.__class__.__name__ == 'MPLayer':
                module.gaps_var = []
                module.gaps_mean = []

    return model


def set_MPLayer_val_sim_len_for_eval(model, eval_sim_len):
    with torch.no_grad():
        for name, module in model._modules.items():
            if hasattr(module, "_modules"):
                set_MPLayer_val_sim_len_for_eval(module, eval_sim_len)
            if module.__class__.__name__ == 'MPLayer':
                module.val_sim_len = eval_sim_len

    return model


def set_MPLayer_val_sim_len_for_val(model, val_sim_len):
    with torch.no_grad():
        for name, module in model._modules.items():
            if hasattr(module, "_modules"):
                set_MPLayer_val_sim_len_for_val(module, val_sim_len)
            if module.__class__.__name__ == 'MPLayer':
                module.val_sim_len = val_sim_len

    return model


def single_update_sigma_and_mu(model, sigma_mean_of_layer, mu_mean_of_layer):
    with torch.no_grad():
        for name, module in model._modules.items():
            if hasattr(module, "_modules"):
                single_update_sigma_and_mu(module, sigma_mean_of_layer, mu_mean_of_layer)
            if module.__class__.__name__ == 'MPLayer':
                module.sigma = sigma_mean_of_layer
                module.mu = mu_mean_of_layer

    return model


def piecewise_update_sigma(model, sigmas):
    i = 0
    with torch.no_grad():
        for name, module in model.named_modules():
            if hasattr(module, 'up'):
                module.sigma = sigmas[i]
                # module.mu = mus[i]
                i += 1
    return model


def piecewise_update_up(model, up):
    i = 0
    with torch.no_grad():
        for name, module in model.named_modules():
            if hasattr(module, 'up'):
                module.up = up[i]
                # module.mu = mus[i]
                i += 1
    return model


def update_pre2_mu_and_sigma_for_resnet(model, sigmas, mus):
    i = 0
    with torch.no_grad():
        for name, module in model.named_modules():
            if hasattr(module, 'up'):
                if i >= 2:
                    module.pre2_mu = mus[i - 2]
                    module.pre2_sigma = sigmas[i - 2]
                i += 1
    return model


def set_MPLayer_snn_mode(model, snn):
    with torch.no_grad():
        for name, module in model._modules.items():
            if hasattr(module, "_modules"):
                set_MPLayer_snn_mode(module, snn)
            if module.__class__.__name__ == 'MPLayer':
                module.snn_mode = snn
        return model


def set_MPLayer_train_mode(model, train_mode):
    with torch.no_grad():
        for name, module in model._modules.items():
            if hasattr(module, "_modules"):
                set_MPLayer_train_mode(module, train_mode)
            if module.__class__.__name__ == 'MPLayer':
                module.train_mode = train_mode
        return model


def update_statistics(mean, M2, n, new_data):
    for x in new_data:
        n += 1
        delta = x - mean
        mean += delta / n
        delta2 = x - mean
        M2 += delta * delta2
    return mean, M2, n


def calculate_var(mean, M2, n):
    if n < 2:
        return 0.0  # Standard deviation requires at least 2 data points
    variance = M2 / (n - 1)
    return variance


def calculate_mid_variables(total_mean, total_M2, total_n, batch):
    total_mean, total_M2, total_n = update_statistics(total_mean, total_M2, total_n, batch)
    return total_mean, total_M2, total_n


def flag_activation_layer_in_basicblock(model):
    for name, module in model.named_modules():
        if hasattr(module, 'residual_function'):
            residual_function = module.residual_function
            for sub_name, sub_module in residual_function.named_modules():
                if isActivation(sub_module.__class__.__name__.lower()):
                    sub_module.is_basicblock_activation_layer = True
    return model
