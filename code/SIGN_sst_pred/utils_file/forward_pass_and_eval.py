from collections import defaultdict
from ctypes import util
import re
import time
import torch
from tqdm import tqdm
import random
from utils_file import arg_parser, data_loader
import numpy as np
from model.modules import *
from model import utils, model_loader
from torch_geometric.utils import degree
import copy
import pandas as pd

def forward_pass_and_eval(
    args,
    decoder,
    batchs,
    epoch,
    c_mask = None,
    f_mask = None,
    save = False
):
    start = time.time()
    losses = defaultdict(lambda: torch.zeros((), device=args.device.type))
    steps = 0


    device = args.device
    batchs = batchs.to(device)
    data, edge_index, batch, t = batchs.x, batchs.edge_index, batchs.batch, batchs.t.reshape(-1,args.time_stamp)[0]
    if save:
        args.teacher = 120


    data = data.to(device)
    edge_index = edge_index.to(device)
    batch = batch.to(device)
    t = t.to(device)
    batch_size = batch.max()
    if len(data.shape) == 2:
        data = data.unsqueeze(2)
    target = data[:, 1:, [args.k]]


    if args.decoder == 'CGSI':
        input_batch = copy.deepcopy(batchs)
        data = data[:,::10,:]
        t = t[::10]
        input_batch.t = t
        input_batch.x = data
        target = data
    else:
        input_batch = batchs


    if args.decoder is not None:
        output,wc, wf = decoder(
            t,
            input_batch,
            c_mask = c_mask,
            f_mask = f_mask
        )

    if save:
        if args.ode_model == 'enso':
            true_data = target[:, :, 0].cpu().detach().numpy().T  # shape: (N, M)

            #output2,_,_ = decoder(t, input_batch, c_mask = c_mask,f_mask = f_mask)
            pred_data = output[:, :, 0].cpu().detach().numpy().T  # shape: (N, M)

            true_df = pd.DataFrame(true_data)
            pred_df = pd.DataFrame(pred_data)
            true_df.to_csv(f'true_{args.ode_model}_dim_{args.k}.csv', index=False, header=False)
            pred_df.to_csv(f'pred_{args.ode_model}_dim_{args.k}.csv', index=False, header=False)
            xy = batchs.xy.cpu().detach().numpy()
            np.savetxt('enso_xy.csv', xy, delimiter=',')
        else:
            true_data = target[:1000, :, 0].cpu().detach().numpy().T  # shape: (N, M)

            #output2,_,_ = decoder(t, input_batch, c_mask = c_mask,f_mask = f_mask)
            pred_data = output[:1000, :, 0].cpu().detach().numpy().T  # shape: (N, M)

            true_df = pd.DataFrame(true_data)
            pred_df = pd.DataFrame(pred_data)
            true_df.to_csv(f'true_{args.ode_model}_dim_{args.k}.csv', index=False, header=False)
            pred_df.to_csv(f'pred_{args.ode_model}_dim_{args.k}.csv', index=False, header=False)


    losses['loss_wc'] = utils.l1_loss(wc)
    losses['loss_wf'] = utils.l1_loss(wf)
    losses["loss_mse"] = F.mse_loss(output, target)
    losses["loss_mape"] = utils.MAPE(output, target)
    losses["loss"] = losses["loss_mse"]
    losses["inference time"] = time.time() - start
    losses['wc'] = wc
    losses['wf'] = wf
    return losses

if __name__ == '__main__':
    pass


