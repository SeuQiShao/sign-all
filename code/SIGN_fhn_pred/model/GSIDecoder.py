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

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

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
        self.D = args.dims


        f_basis = utils.fun_lib(torch.empty(0, args.dims), self.poly_p, self.poly_n, activate=self.activate, device="cpu", names=True)
        c_basis = utils.coupled_fun_lib(None, None, self.poly_p, self.poly_n, activate=self.activate, device="cpu", names=True)
        self.Mf = len(f_basis)
        self.Mc = len(c_basis)


        self.wf_2 = nn.Parameter(
            torch.cat(((0.9 + 0.1 * torch.rand(self.Mf, self.D)),
                       (0.1 * torch.rand(self.Mf, self.D))), dim=0),
            requires_grad=True
        )
        self.wc_2 = nn.Parameter(
            torch.cat(((0.9 + 0.1 * torch.rand(self.Mc, self.D)),
                       (0.1 * torch.rand(self.Mc, self.D))), dim=0),
            requires_grad=True
        )


        self.UseEdgeAttr = getattr(args, "UseEdgeAttr", False)
        if self.UseEdgeAttr:
            self.edge_attr_all = nn.Parameter(torch.empty(self.num_nodes, self.num_nodes))
            nn.init.xavier_uniform_(self.edge_attr_all)


        self.memory_efficient = getattr(args, "memory_efficient", True)

        self.chunk_size = getattr(args, "chunk_size", None)

    def forward(self, t, x, batchs):


        device = x.device
        edge_index = batchs.edge_index
        k = getattr(batchs, "k", None)


        if self.UseEdgeAttr:
            edge_attr = self.edge_attr_all[edge_index[0] % self.num_nodes, edge_index[1] % self.num_nodes].view(-1, 1)
        else:
            edge_attr = 1.0


        c_out = self.propagate(edge_index, x=x, edge_attr=edge_attr, wc_1=batchs.c_mask, k=k)


        f_mask = batchs.f_mask.to(device)  # [Mf, D]

        N = x.shape[0]
        D = self.D
        f_out = torch.zeros(N, D, device=device)


        f_active_idx = []  # list of tensors of selected indices for each d
        f_mask_vals = []   # list of tensors of mask values for selected indices per d
        for d in range(D):
            col = f_mask[:, d]
            sel = torch.nonzero(col.abs() > 0, as_tuple=True)[0]
            if sel.numel() == 0:
                f_active_idx.append(None)
                f_mask_vals.append(None)
            else:
                f_active_idx.append(sel)
                f_mask_vals.append(col[sel])


        for d in range(D):
            sel = f_active_idx[d]
            if sel is None:
                continue


            mask_full = torch.zeros(self.Mf, device=device)
            mask_full[sel] = 1.0

            with torch.no_grad():


                F_basis = utils.fun_lib(x, self.poly_p, self.poly_n, self.device, self.activate, mask=mask_full)
                # F_basis: [N, r]
                if F_basis.numel() == 0:
                    continue
                F_basis_pm = torch.cat([F_basis, -F_basis], dim=1)  # [N, 2*r]


            pos = sel
            neg = sel + self.Mf
            posneg = torch.cat([pos, neg], dim=0)


            mask_vals = f_mask_vals[d]
            mask_pm = torch.cat([mask_vals, mask_vals], dim=0).to(device)  # [2*r]

            wf_cols = self.wf_2[posneg, d].to(device)  # [2*r]
            weights = mask_pm * wf_cols  # [2*r]


            f_out[:, d] = F_basis_pm @ weights

        return c_out + f_out

    def message(self, x_i, x_j, edge_attr, wc_1, k):


        device = x_i.device
        E = x_i.shape[0]
        D = self.D


        wc = wc_1.to(device)
        wc_active_idx = []
        wc_mask_vals = []
        for d in range(D):
            col = wc[:, d]
            sel = torch.nonzero(col.abs() > 0, as_tuple=True)[0]
            if sel.numel() == 0:
                wc_active_idx.append(None)
                wc_mask_vals.append(None)
            else:
                wc_active_idx.append(sel)
                wc_mask_vals.append(col[sel])

        out = torch.zeros(E, D, device=device)


        for d in range(D):
            sel = wc_active_idx[d]
            if sel is None:
                continue


            mask_full = torch.zeros(self.Mc, device=device)
            mask_full[sel] = 1.0

            with torch.no_grad():

                C_basis = utils.coupled_fun_lib(x_i[:, d:d+1], x_j[:, d:d+1], self.poly_p, self.poly_n, self.device, self.activate, mask=mask_full)
                if C_basis.numel() == 0:
                    continue
                C_basis_pm = torch.cat([C_basis, -C_basis], dim=1)  # [E, 2*r]


            pos = sel
            neg = sel + self.Mc
            posneg = torch.cat([pos, neg], dim=0)

            mask_vals = wc_mask_vals[d]
            mask_pm = torch.cat([mask_vals, mask_vals], dim=0).to(device)  # [2*r]
            wc_cols = self.wc_2[posneg, d].to(device)  # [2*r]
            weights = mask_pm * wc_cols  # [2*r]


            c_d = C_basis_pm @ weights


            if not torch.is_tensor(edge_attr):
                ea = torch.ones(E, device=device) * float(edge_attr)
            else:
                ea = edge_attr.view(-1).to(device)

            out[:, d] = ea * c_d

        return out

    def update(self, aggr_out):
        return aggr_out


#                       (0.1 * torch.rand(self.num_func_lib,1))), dim=0),

#         )


#                      (0.1 * torch.rand(self.num_coupled_fun_lib,1))), dim=0),

#         )


#             )


#                                         edge_index[1]%self.num_nodes].view(-1,1)


#             edge_index,


#         )


#                                     self.device, self.activate, mask = wf_1.detach().squeeze())


#                                          self.device, self.activate, mask = wc_1.detach().squeeze())


class DGSIDecoder(nn.Module):
    def __init__(self, args):
        super(DGSIDecoder, self).__init__()
        self.teacher = args.teacher
        self.time_stamp = args.time_stamp
        self.activate = args.activate
        self.device = args.device
        self.dims = args.dims


        self.GSICell = GSICell(args)
        self.num_func_lib = self.GSICell.Mf
        self.num_coupled_fun_lib = self.GSICell.Mc


    def single_step_forward(self, t, batchs, step_x):

        x_dot = self.GSICell(t, step_x, batchs)

        return x_dot * t + step_x


    def forward(self, t, batchs, c_mask =None, f_mask =None):
        out = []
        if c_mask is not None:
            batchs.c_mask = c_mask.to(self.device)
            batchs.f_mask = f_mask.to(self.device)
        else:
            batchs.c_mask = torch.ones(self.num_coupled_fun_lib,self.dims).to(self.device)
            batchs.f_mask = torch.ones(self.num_func_lib,self.dims).to(self.device)
        total_steps = batchs.x.shape[1]
        start_step = 1
        if batchs.train:
            start_step = torch.randint(0, total_steps - 200, (1,)).item()
            for i in range(start_step, start_step + 200):
                if i == start_step:
                    step_x = batchs.x[:,i,:]
                else:
                    step_x = out[-1]
                out.append(self.single_step_forward(torch.diff(t)[0], batchs, step_x))
        else:
            for i in range(total_steps - 1):
                if i%self.teacher == 0:
                    step_x = batchs.x[:,i,:]
                else:
                    step_x = out[-1]
                out.append(self.single_step_forward(torch.diff(t)[0], batchs, step_x))

        out = torch.stack(out,1)

        wc_2 = self.GSICell.wc_2.squeeze()
        wf_2 = self.GSICell.wf_2.squeeze()
        wc = -wc_2.reshape(2,-1, out.shape[-1]).T.diff().squeeze().T * batchs.c_mask.squeeze()
        wf = -wf_2.reshape(2,-1,out.shape[-1]).T.diff().squeeze().T * batchs.f_mask.squeeze()

        return out, wc, wf, start_step


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
        self.time_stamp = args.time_stamp
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


