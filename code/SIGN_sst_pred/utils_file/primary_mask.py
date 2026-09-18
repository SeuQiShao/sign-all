from tqdm import tqdm
from utils_file import arg_parser, data_loader
from model.modules import *
from model import utils, model_loader
import random
import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit, Lasso, ARDRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.feature_selection import mutual_info_regression
from joblib import Parallel, delayed
from collections import defaultdict
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNetCV
from collections import Counter
from sklearn.cluster import DBSCAN

def lasso_AIC(model, x, y):
    mse = np.mean((model.predict(x) - y)**2)
    if hasattr(model, 'intercept_'):
        p = (np.abs(model.coef_) > 0).sum() + np.sum(np.abs(model.intercept_) > 0)
    else:
        p = (np.abs(model.coef_) > 0).sum()
    aic = x.shape[0] * np.log(mse) + 2 * p

    return aic

def compute_node_features(data, i, edge_index, t, args, device):
    neighbor_i = edge_index[0][edge_index[1] == i]
    neighbor_num = len(neighbor_i)


    neighbor_index = neighbor_i
    neighbor_coef = 1

    x_neighbor = data[neighbor_index, :, args.k]
    x_i = data[i, :, args.k]

    if neighbor_num == 0:

        print(f"Node {i} has no neighbors, skipping.")
        x_c = None
    else:

        x_c = 0
        for j in x_neighbor:
            x_c += utils.coupled_fun_lib(x_i.reshape(-1, 1), j.reshape(-1, 1),
                                         args.poly_p, args.poly_n, device, activate=args.activate)
        if args.agg == 'mean':
            x_c = x_c / x_neighbor.shape[0]
        x_c = x_c * neighbor_coef


    x_vals = data[i, :, args.k].cpu().numpy()
    t_vals = t.cpu().numpy()
    dt = t_vals[1] - t_vals[0]


    x1 = utils.fun_lib(data[i, :, args.k].reshape(-1, 1), args.poly_p, args.poly_n, device, activate=args.activate)


    if args.t_basis:
        x_t = utils.t_fun_lib(t,args.T_max_k, device)
        x1 = torch.cat((x1, x_t), 1)


    if x_c is not None:

        x_data = torch.cat((x1[:,1:], x_c), 1)
    else:
        x_data = x1[:,1:]

    if args.five_points:
        x_dot_vals = (-x_vals[4:] + 8 * x_vals[3:-1] - 8 * x_vals[1:-3] + x_vals[:-4]) / (12 * dt)
        x_data = x_data[2:-2, :]

    else:
        x_dot_vals = (x_vals[1:] - x_vals[:-1])/ dt
        x_data = x_data[:-1, :]
    x_dot = torch.tensor(x_dot_vals, device=device)

    return x_data, x_dot

def lasso_fit_single_node(x_data, x_dot, alpha=0.01):

    model1 = Lasso(alpha = 0.025, fit_intercept=True)
    model2 = OrthogonalMatchingPursuit(n_nonzero_coefs = 10, fit_intercept=True)


    model1.fit(x_data.cpu(), x_dot.cpu())
    model2.fit(x_data.cpu(), x_dot.cpu())


    #        ]


    model = [model1, model2]
    aic = []
    for i in model:
        aic.append(lasso_AIC(i, x_data.cpu(), x_dot.cpu().numpy()))

    model_index = np.argmin(aic)
    coef = model[model_index].coef_
    coef[np.abs(coef) < 0.001] = 0
    return coef, model[model_index].intercept_


def summarize_basis_functions(coefs_list):

    coefs_matrix = np.array(coefs_list)
    basis_indicator = (np.abs(coefs_matrix) > 1e-3).any(axis=0).astype(int)
    return basis_indicator


def generate_primary_mask(args, batchs):
    print('Start Lasso mask...')
    device = args.device
    batchs = batchs.to(device)
    fun_names = utils.fun_lib(torch.empty((0,1)), args.poly_p, args.poly_n, device="cpu", activate=args.activate, names=True)
    if args.t_basis:
        t_names = utils.t_fun_lib(torch.empty((0,1)), args.T_max_k, device="cpu", names=True)
        fun_names = fun_names + t_names
    coupled_names = utils.coupled_fun_lib(None, None, args.poly_p, args.poly_n, device="cpu", activate=args.activate, names=True)
    f_num = len(fun_names)
    c_num = len(coupled_names)

    print(batchs)
    time_stamp = batchs.x.shape[1]
    data, edge_index, batch, t = batchs.x, batchs.edge_index, batchs.batch, batchs.t.reshape(-1,args.time_stamp)[0]
    nums = min(args.lasso_node_num, data.shape[0])
    random_index = random.sample(list(np.arange(data.shape[0])), int(nums))

    coefs = []
    intercepts = []
    count = 0
    x_data = []
    x_dot = []
    p_num = []
    for i in random_index:
        x_data0, x_dot0 = compute_node_features(data, i, edge_index, t[:time_stamp], args, device)
        if x_data0.shape[1] < f_num:
            continue
        coef0, intercept0= lasso_fit_single_node(x_data0, x_dot0)

        coefs.append(np.hstack((coef0, intercept0)))
        intercepts.append(intercept0)
        p_num.append(np.sum(np.abs(coef0)>0))
        x_dot.append(x_dot0)
        x_data.append(x_data0)
    vectors_array = np.array(coefs)
    normal_coef = (vectors_array - np.mean(vectors_array, 0))/(np.std(vectors_array, 0) + 1e-5)
    dbscan = DBSCAN(eps=0.1, min_samples=nums//10)
    dbscan.fit(normal_coef)


    selected_indices = [i for i, p in enumerate(dbscan.labels_) if p >= 0]
    print('core node:', len(selected_indices))
    if len(selected_indices) == 0:
        selected_indices = np.arange(len(x_data)).tolist()


    filtered_x_data = [x_data[i] for i in selected_indices]
    filtered_x_dot = [x_dot[i] for i in selected_indices]


    if filtered_x_data:
        x_data_aggregated = torch.cat(filtered_x_data, dim=0)
        x_dot_aggregated = torch.cat(filtered_x_dot, dim=0)
    else:
        x_data_aggregated = None
        x_dot_aggregated = None
    coef, intercept= lasso_fit_single_node(x_data_aggregated, x_dot_aggregated)


    if True:
        c_mask = torch.zeros(c_num, 1)
        f_mask = torch.zeros(f_num, 1)
        coef = torch.tensor(coef, dtype=torch.float32, requires_grad=False)
        intercept = torch.tensor(intercept, dtype=torch.float32, requires_grad=False)
        f_mask[0, 0] = intercept
        f_mask[1:,0] = coef[:(f_num-1)]
        c_mask[:,0] = coef[(f_num-1):]

        expression = utils.functions(args, args.poly_p, args.poly_n, f_mask, c_mask, activate=args.activate)[0]
        print('basis:'.format(i), expression)
    return f_mask, c_mask

