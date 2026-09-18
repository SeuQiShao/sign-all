import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from model import utils
from model.modules import *
import warnings
from torch_geometric.nn import MessagePassing
warnings.filterwarnings("ignore")
import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint, odeint


import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class GSICell(MessagePassing):
    def __init__(self, args):

        super(GSICell, self).__init__(aggr=args.agg)
        self.args = args


        self.poly_p = args.poly_p
        self.poly_n = args.poly_n
        self.activate = args.activate
        self.device = args.device
        self.num_nodes = args.num_atoms
        self.t_basis = args.t_basis
        self.t_max_k = args.T_max_k
        fun_names = utils.fun_lib(torch.empty((0,1)), self.poly_p, self.poly_n, device="cpu", activate=self.activate, names=True)

        if args.t_basis:
            t_names = utils.t_fun_lib(torch.empty((0,1)), args.T_max_k, device="cpu", names=True)
            self.num_t_lib = len(t_names)
            fun_names = fun_names + t_names
        else:
            self.num_t_lib = 0

        coupled_names = utils.coupled_fun_lib(None, None, self.poly_p, self.poly_n, device="cpu", activate=self.activate, names=True)
        self.num_func_lib = len(fun_names)
        self.num_coupled_fun_lib = len(coupled_names)


        #     self.num_func_lib += 3

        self.wf_2 = nn.Parameter(
            torch.cat((
                0.9 + 0.1 * torch.rand(self.num_func_lib, 1),
                0.1 * torch.rand(self.num_func_lib, 1)
            )),
            requires_grad=True
        )


        #     self.num_coupled_fun_lib += 12

        self.wc_2 = nn.Parameter(
            torch.cat((
                0.9 + 0.1 * torch.rand(self.num_coupled_fun_lib, 1),
                0.1 * torch.rand(self.num_coupled_fun_lib, 1)
            )),
            requires_grad=True
        )

        self.UseEdgeAttr = args.UseEdgeAttr
        if self.UseEdgeAttr:
            self.edge_attr_all = nn.Parameter(
                torch.randn(self.num_nodes, self.num_nodes)
            )
            nn.init.xavier_uniform_(self.edge_attr_all)

    def forward(self, t, x, batchs):

        edge_index = batchs.edge_index


        wc_1 = batchs.c_mask


        if self.UseEdgeAttr:
            edge_attr = self.edge_attr_all[edge_index[0]%self.num_nodes,
                                        edge_index[1]%self.num_nodes].view(-1,1)
        else:
            edge_attr = 1.0


        c_out = self.propagate(
            edge_index,
            x=x,
            edge_attr=edge_attr,
            wc_1=wc_1
        )


        wf_1 = batchs.f_mask


        with torch.no_grad():
            F_msg = utils.fun_lib(x, self.poly_p, self.poly_n,
                                    self.device, self.activate, mask = wf_1[:self.num_func_lib-self.num_t_lib].detach().squeeze())

            if self.t_basis:
                t_msg = utils.t_fun_lib(t,self.t_max_k, self.device, mask = wf_1[-self.num_t_lib:].detach().squeeze())
                t_msg = t_msg[0].repeat(F_msg.shape[0],1)
                F_msg = torch.cat((F_msg, t_msg), 1)
            F_msg = torch.cat([F_msg, -F_msg], dim=1)
        f_mask_index = (wf_1.abs() > 0)
        wf_1 = wf_1[f_mask_index]
        if wf_1.dim() > 0:
            wf_1 = torch.cat([wf_1, wf_1])
        extended_mask = torch.cat([f_mask_index, f_mask_index], dim=0)
        F_weights = wf_1 * self.wf_2[extended_mask]
        f_out = torch.mm(F_msg, F_weights.unsqueeze(1))

        return c_out + f_out


    def message(self, x_i, x_j, edge_attr, wc_1):


        with torch.no_grad():
            C_msg = utils.coupled_fun_lib(x_i, x_j, self.poly_p, self.poly_n,
                                         self.device, self.activate, mask = wc_1.detach().squeeze())

            C_msg = torch.cat([C_msg, -C_msg], dim=1)


        mask_index = (wc_1.abs() > 0)
        wc_1 = wc_1[mask_index]
        if wc_1.dim() > 0:
            wc_1 = torch.cat([wc_1, wc_1])
        extended_mask = torch.cat([mask_index, mask_index], dim=0)
        C_weights = wc_1 * self.wc_2[extended_mask]


        return edge_attr * torch.mm(C_msg, C_weights.unsqueeze(1))

    def update(self, aggr_out):

        return aggr_out


class DGSIDecoder(nn.Module):
    def __init__(self, args):
        super(DGSIDecoder, self).__init__()
        self.teacher = args.teacher

        self.activate = args.activate
        self.device = args.device
        self.k = args.k

        self.GSICell = GSICell(args)
        self.num_func_lib = self.GSICell.num_func_lib
        self.num_coupled_fun_lib = self.GSICell.num_coupled_fun_lib


    def single_step_forward(self, t, batchs, step_x):

        x_dot = self.GSICell(t, step_x, batchs)

        return x_dot * torch.diff(batchs.t)[0] + step_x


    def forward(self, t, batchs, c_mask =None, f_mask =None):
        out = []
        batchs.k = self.k
        time_stamp = batchs.x.shape[1]
        if c_mask is not None:
            batchs.c_mask = c_mask.to(self.device)
            batchs.f_mask = f_mask.to(self.device)
        else:
            batchs.c_mask = torch.ones(self.num_coupled_fun_lib,1).to(self.device)
            batchs.f_mask = torch.ones(self.num_func_lib,1).to(self.device)
        for i in range(time_stamp - 1):
            if i%self.teacher == 0:
                step_x = batchs.x[:,i,:]
            else:
                step_x = out[-1]
            out.append(self.single_step_forward(t[[i]], batchs, step_x))

        out = torch.stack(out,1)

        wc_2 = self.GSICell.wc_2.squeeze()
        wf_2 = self.GSICell.wf_2.squeeze()
        wc = -wc_2.reshape(2,-1).T.diff().squeeze() * batchs.c_mask.squeeze()
        wf = -wf_2.reshape(2,-1).T.diff().squeeze() * batchs.f_mask.squeeze()

        return out, wc, wf


class ParametricODE(nn.Module):

    def __init__(self, odefunc):
        super().__init__()
        self.odefunc = odefunc
        self.current_batchs = None

    def forward(self, t, x):

        if isinstance(t, torch.Tensor) and t.dim() > 0:

            return self.odefunc(t[0], x, self.current_batchs)
        else:
            return self.odefunc(t, x, self.current_batchs)

    def set_batchs(self, batchs):

        self.current_batchs = batchs


class ODEBlock(nn.Module):
    def __init__(self, odefunc, rtol=1e-3, atol=1e-4, method='dopri5', adjoint=False):
        super().__init__()
        self.odefunc = odefunc
        self.rtol = rtol
        self.atol = atol
        self.method = method
        self.adjoint = adjoint

    def forward(self, vt, x, batchs):

        self.odefunc.set_batchs(batchs)


        integration_time = vt.type_as(x)


        if self.adjoint:
            solution = odeint_adjoint(
                self.odefunc,
                x,
                integration_time,
                rtol=self.rtol,
                atol=self.atol,
                method=self.method
            )
        else:
            solution = odeint(
                self.odefunc,
                x,
                integration_time,
                rtol=self.rtol,
                atol=self.atol,
                method=self.method
            )


        return solution


class CGSIDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.gsicell = GSICell(args)
        self.parametric_ode = ParametricODE(self.gsicell)


        self.neural_dynamic_layer = ODEBlock(
            self.parametric_ode,


        )


        self.teacher = args.teacher

        self.device = args.device


        self.register_buffer('c_mask', torch.ones(self.gsicell.num_coupled_fun_lib, 1))
        self.register_buffer('f_mask', torch.ones(self.gsicell.num_func_lib, 1))

    def forward(self, t, batchs, c_mask=None, f_mask=None):


        if c_mask is not None:
            batchs.c_mask = c_mask.to(self.device)
            batchs.f_mask = f_mask.to(self.device)
        else:
            batchs.c_mask = self.c_mask
            batchs.f_mask = self.f_mask


        all_preds = []
        total_steps = batchs.x.shape[1]
        teacher_interval = self.teacher


        num_windows = (total_steps + teacher_interval - 1) // teacher_interval

        for window_idx in range(num_windows):

            start_step = window_idx * teacher_interval
            end_step = min((window_idx + 1) * teacher_interval, total_steps)


            window_times = t[start_step:end_step]


            if len(window_times) < 2:
                continue


            if window_idx == 0:

                pred_times = 0
                x0 = batchs.x[:, start_step, :]
            else:

                x0 = batchs.x[:, pred_times, :].detach()


            window_pred = self.neural_dynamic_layer(
                vt=window_times,
                x=x0,
                batchs=batchs
            )


            if window_idx == 0:
                all_preds.append(window_pred)
            else:
                all_preds.append(window_pred)


            pred_times = pred_times + len(window_times)


        full_pred = torch.cat(all_preds, dim=0)


        output = full_pred.permute(1, 0, 2)


        wc_2 = self.gsicell.wc_2.squeeze()
        wf_2 = self.gsicell.wf_2.squeeze()
        wc = -wc_2.reshape(2, -1).T.diff().squeeze() * batchs.c_mask.squeeze()
        wf = -wf_2.reshape(2, -1).T.diff().squeeze() * batchs.f_mask.squeeze()

        return output, wc, wf


if __name__ == '__main__':
    pass


