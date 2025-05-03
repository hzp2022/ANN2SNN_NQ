import pickle
from torch import nn
import torch
from tqdm import tqdm
from utils import *
import numpy as np
import torch.optim as optim
import gc
import pandas as pd
import math


def mp_test(test_dataloader, model, net_arch, presim_len, sim_len, device):
    new_tot = torch.zeros(sim_len).cuda(device)
    model = model.cuda(device)
    model.eval()

    with torch.no_grad():
        for img, label in tqdm(test_dataloader):
            new_spikes = 0
            img = img.cuda(device)
            label = label.cuda(device)

            for t in range(presim_len + sim_len):
                out = model(img)

                if t >= presim_len:
                    new_spikes += out
                    new_tot[t - presim_len] += (label == new_spikes.max(1)[1]).sum().item()

    return new_tot


def train_ann(train_dataloader, val_dataloader, test_dataloader, model, epochs, lr, wd, device, save_name, val_sim_len,
              best_eval_snn_sim_len):
    model = model.cuda(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_fn = nn.CrossEntropyLoss()
    best_acc = 0
    best_snn_acc = 0
    gap_mean_of_layers = []
    gap_std_of_layers = []
    ann_accuracy = []
    snn_accuracy = []
    sigmas = 0.
    mus = 0.
    up = [0] * 19

    layer_len = 0
    for name, moudle in model.named_modules():
        if hasattr(moudle, 'up'):
            layer_len += 1

    for epoch in range(epochs):
        epoch_loss = 0
        lenth = 0

        model.train()
        set_MPLayer_train_mode(model, True)
        set_MPLayer_snn_mode(model, False)
        for img, label in tqdm(train_dataloader):
            img = img.cuda(device)
            label = label.cuda(device)
            optimizer.zero_grad()
            out = model(img)
            loss = loss_fn(out, label)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            lenth += len(img)

        acc = eval_ann(test_dataloader, model, device)
        ann_accuracy.append(round(acc * 0.01, 2))

        if acc > best_acc:
            best_acc = acc

        set_MPLayer_val_sim_len_for_eval(model, best_eval_snn_sim_len)

        snn_acc = eval_snn_for_save_best_snn_model(test_dataloader, model, device, best_eval_snn_sim_len)

        set_MPLayer_val_sim_len_for_val(model, val_sim_len)

        save_snn_acc = snn_acc.item()
        snn_accuracy.append(round(save_snn_acc * 0.01, 2))

        if snn_acc > best_snn_acc:
            best_snn_acc = snn_acc
            torch.save(model.state_dict(), save_name + '.pth')
            with open('sigma_' + save_name + '.pkl', 'wb') as f:
                pickle.dump(sigmas, f)
            with open('up_' + save_name + '.pkl', 'wb') as f:
                pickle.dump(up, f)

        scheduler.step()

        # Validation
        model.eval()
        var_of_gap_std = [0] * layer_len
        var_of_gap_mean = [0] * layer_len
        set_MPLayer_train_mode(model, False)

        var_total_mean = [0] * layer_len
        var_total_M2 = [0] * layer_len
        var_total_n = [0] * layer_len

        mean_total_mean = [0] * layer_len
        mean_total_M2 = [0] * layer_len
        mean_total_n = [0] * layer_len

        with torch.no_grad():
            for img, label in tqdm(val_dataloader):
                img = img.cuda(device)
                label = label.cuda(device)

                set_MPLayer_snn_mode(model, True)
                out = model(img)

                set_MPLayer_snn_mode(model, False)
                out = model(img)

                # Welford algorithm calculates the mean and standard deviation
                i = 0
                for name, moudle in model.named_modules():
                    if hasattr(moudle, 'up'):
                        var_total_mean[i], var_total_M2[i], var_total_n[i] = update_statistics(var_total_mean[i],
                                                                                               var_total_M2[i],
                                                                                               var_total_n[i],
                                                                                               moudle.gaps_var)
                        mean_total_mean[i], mean_total_M2[i], mean_total_n[i] = update_statistics(mean_total_mean[i],
                                                                                                  mean_total_M2[i],
                                                                                                  mean_total_n[i],
                                                                                                  moudle.gaps_mean)
                        i += 1

                reset_gaps_var_and_mean(model)

        aver_gap_mean_of_layer = 0.
        var_of_gap_mean_of_all_layer = 0.
        aver_gap_std_of_layer = 0.
        var_of_gap_var_of_all_layer = 0.
        the_gap_std_of_each_layer = [0] * layer_len
        the_gap_mean_of_each_layer = [0] * layer_len

        float_list_of_mean_total_mean = [item.item() for item in mean_total_mean]
        float_list_of_var_total_mean = [item.item() for item in var_total_mean]

        for j in range(layer_len):
            aver_gap_mean_of_layer += mean_total_mean[j]
            the_gap_mean_of_each_layer[j] = float_list_of_mean_total_mean[j]

            aver_gap_std_of_layer += var_total_mean[j]
            the_gap_std_of_each_layer[j] = math.sqrt(float_list_of_var_total_mean[j])

        print(f'ANNs training Epoch {epoch}: Val_loss: {epoch_loss / lenth}' ' acc: {:.2f}'.format(acc * 0.01),
              ' Best_acc: {:.2f}'.format(best_acc * 0.01), ' snn_acc: {:.2f}'.format(snn_acc * 0.01),
              ' Best_snn_acc: {:.2f}'.format(best_snn_acc * 0.01))

        aver_gap_mean_of_layer = aver_gap_mean_of_layer / layer_len

        gap_mean_of_layers.append(the_gap_mean_of_each_layer)

        aver_gap_std_of_layer = aver_gap_std_of_layer / layer_len
        aver_gap_std_of_layer = math.sqrt(aver_gap_std_of_layer)
        aver_gap_std_of_layer = aver_gap_std_of_layer

        gap_std_of_layers.append(the_gap_std_of_each_layer)

        mus = the_gap_mean_of_each_layer
        sigmas = the_gap_std_of_each_layer

        i = 0
        for name, moudle in model.named_modules():
            if hasattr(moudle, 'up'):
                up[i] = moudle.up
                i += 1

        piecewise_update_sigma(model, the_gap_std_of_each_layer)

    return model


def eval_ann(test_dataloader, model, device):
    set_MPLayer_train_mode(model, True)
    set_MPLayer_snn_mode(model, False)
    tot = 0.
    model.eval()
    model.cuda(device)

    with torch.no_grad():
        for img, label in tqdm(test_dataloader):
            img = img.cuda(device)
            label = label.cuda(device)
            out = model(img)
            tot += (label == out.max(1)[1]).sum().item()

    return tot


def eval_snn(test_dataloader, model, sim_len, device):
    tot = torch.zeros(sim_len).cuda(device)
    model = model.cuda(device)
    model.eval()
    with torch.no_grad():
        for img, label in tqdm(test_dataloader):
            spikes = 0
            img = img.cuda(device)
            label = label.cuda(device)
            for t in range(sim_len):
                out = model(img)
                spikes += out
                tot[t] += (label == spikes.max(1)[1]).sum().item()
            reset_net(model)

    return tot


def eval_snn_for_save_best_snn_model(test_dataloader, model, device, eval_snn_sim_len):
    set_MPLayer_snn_mode(model, True)
    tot = torch.zeros(eval_snn_sim_len).cuda(device)
    model.eval()
    model.cuda(device)

    with torch.no_grad():
        for img, label in tqdm(test_dataloader):
            spikes = 0
            img = img.cuda(device)
            label = label.cuda(device)
            out = model(img)
            out = out.reshape(eval_snn_sim_len, out.shape[0] // eval_snn_sim_len, out.shape[1])
            for t in range(eval_snn_sim_len):
                spikes += out[t]
                tot[t] += (label == spikes.max(1)[1]).sum().item()
            reset_MPLayer_neuron(model)

    return tot[eval_snn_sim_len - 1]
