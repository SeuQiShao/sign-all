import argparse
import torch
import datetime
import numpy as np
from pathlib import Path


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "1", "y"):
        return True
    if v.lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=3, help="Random seed.")
    parser.add_argument(
        "--GPU_to_use", type=int, default=None, help="GPU to use for training"
    )


    parser.add_argument(
        "--poly_p", type=int, default=4, help="Polynomial of Library."
    )
    parser.add_argument(
        "--poly_n", type=int, default=2, help="Neg_Polynomial of Library."
    )
    parser.add_argument(
        "--activate", type=str2bool, default=False, help="activation of Library."
    )


    parser.add_argument("--teacher", type=int, default=5, help="add teacher every t.")


    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of epochs to train."
    )
    parser.add_argument(
        "--batch_size", type=int, default=40, help="Number of samples per batch."
    )
    parser.add_argument("--lam_c", type=float, default=1, help="lambda of w_f, w_c.")
    parser.add_argument("--lam_f", type=float, default=1, help="lambda of w_f, w_c.")
    parser.add_argument("--lam_s", type=float, default=1, help="lambda of similarity.")
    parser.add_argument(
        "--lr", type=float, default=0.005, help="Initial learning rate."
    )
    parser.add_argument(
        "--lr_decay",
        type=int,
        default=20,
        help="After how epochs to decay LR by a factor of gamma.",
    )
    parser.add_argument("--gamma", type=float, default=0.9, help="LR decay factor.")

    parser.add_argument(
        "--lasso_neighbor_num",
        type=int,
        default=200,
        help='lasso_neighbor_num'
    )
    parser.add_argument(
        "--lasso_node_num",
        type=int,
        default=50,
        help='lasso_node_num'
    )

    parser.add_argument(
        "--num_workers", type=int, default=0, help="Number of Workers."
    )
    parser.add_argument('--network', type=str,
                    choices=['random', 'power_law', 'small_world', 'from_file', 'rpg', 'dscm', 'rgm', 'sbm'], default=None)
    parser.add_argument('--ode_model', type=str,
                    choices=['HeatDiffusion', 'Kuramoto', 'SIS', 'Gene', 'MM', 'Mutual', 'FHN', 'HR', 'Rossler', 'Chua'], default='Rossler')
    parser.add_argument('--time_stamp', type=int, default=1000, help="number of timesteps.")
    parser.add_argument(
        '--train_time_points', type=int, default=0,
        help='Uniformly subsample this many points from each full trajectory for training; 0 keeps all points.'
    )
    parser.add_argument('--time_interval', type=float, default=0.01, help="number of sample interval.")
    parser.add_argument('--num_atoms', type=int, default=1000) #37700 1686
    parser.add_argument('--dims', type=int, default=None)
    parser.add_argument('--dataset_seed', type=int, default=0, help='Seed used only for generating/loading the fixed synthetic trajectory.')
    parser.add_argument('--init_scale', type=float, default=None, help='Initial state scale for synthetic data generation.')
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument(
        '--data_root', type=str, default='SIGN-phase/output/data',
        help='Directory that contains generated synthetic data sets.'
    )
    parser.add_argument(
        '--generate_data_if_missing',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Generate the clean synthetic dataset with data/generate_dataset.py when args.root is missing.'
    )
    parser.add_argument(
        '--e1_condition',
        type=str,
        choices=['clean', 'snr50', 'sparse200'],
        default='clean',
        help='E1 Phase-I condition. snr50 adds observation noise to the fixed clean data in memory; sparse200 subsamples the fixed clean trajectory.'
    )
    parser.add_argument('--e1_snr_db', type=float, default=50.0, help='SNR used when --e1_condition snr50.')
    parser.add_argument('--e1_library', type=str, choices=['L1', 'L2', 'L3', 'native'], default='L1', help='Nested E1 candidate library preset.')
    parser.add_argument('--e1_basis_variant', type=str, choices=['legacy', 'trig', 'trig_exp_v2'], default='trig_exp_v2', help='Candidate basis variant. E2_V3 defaults to the final trig_exp_v2 library with self/coupling exponential terms.')
    parser.add_argument('--e1_output_root', type=str, default='SIGN-phase/output/e1', help='Where E1 primary-mask artifacts are saved.')
    parser.add_argument('--e1_run_name', type=str, default='', help='Optional run-name suffix for E1 artifact directories.')
    parser.add_argument(
        '--e1_mask_only',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='For E1, stop after primary_mask and save masks/metrics. Use --no-e1_mask_only to continue SIGN training.'
    )
    parser.add_argument('--e1_dbscan_coef_relative_threshold', type=float, default=0.01, help='Zero coefficients below this fraction of each node coefficient vector max before DBSCAN.')
    parser.add_argument('--e1_dbscan_eps_base', type=float, default=0.15, help='Base DBSCAN eps.')
    parser.add_argument('--e1_dbscan_eps_scale', choices=['fixed', 'sqrt_features'], default='sqrt_features', help='Use eps_base or eps_base * sqrt(number_of_features).')
    parser.add_argument('--e1_dbscan_min_samples', type=int, default=2, help='DBSCAN min_samples.')
    parser.add_argument('--e1_coef_threshold', type=float, default=1e-4, help='Final ARD coefficient threshold used to build support masks.')
    parser.add_argument(
        '--e1_support_selection_threshold',
        type=float,
        default=None,
        help='Threshold used only for reporting selected Phase-I support terms. Defaults to e1_coef_threshold; set to 0 when final Phase-I output should not apply an absolute reporting threshold.'
    )
    parser.add_argument(
        '--e1_final_coef_relative_threshold',
        type=float,
        default=0.0,
        help='Optional final relative coefficient threshold. After final ARD, zero coefficients below this fraction of the max absolute coefficient in the current dimension; 0 disables.'
    )
    parser.add_argument(
        '--e1_max_selected_terms',
        type=int,
        default=0,
        help='Optional cap on final Phase-I selected terms per dimension, ranked by absolute coefficient magnitude across self and coupling terms; 0 disables.'
    )
    parser.add_argument(
        '--e1_second_stage_core_union',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='For the second Phase-I ARD fit, restrict candidate regressors to the union of nonzero first-stage coefficients on DBSCAN core nodes, using pre-DBSCAN coefficients.'
    )
    parser.add_argument('--e1_ard_max_iter', type=int, default=1000, help='ARDRegression max_iter.')
    parser.add_argument('--e1_ard_threshold_lambda', type=float, default=2e5, help='ARDRegression threshold_lambda.')
    parser.add_argument('--e1_lasso_alpha', type=float, default=0.005, help='Lasso alpha for Phase-I Lasso/ARD AIC model selection.')
    parser.add_argument('--e1_lasso_max_iter', type=int, default=2000, help='Lasso max_iter for Phase-I Lasso/ARD AIC model selection.')
    parser.add_argument(
        '--e1_ard_column_scale',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Scale regression columns by their standard deviation before ARD and unscale coefficients afterwards.'
    )
    parser.add_argument(
        '--e1_intercept_mode',
        choices=['aic', 'with', 'without'],
        default='with',
        help='Intercept selection for ARD. with is conservative for Phase-I recall; aic preserves primary_mask model selection for diagnostics.'
    )
    parser.add_argument(
        '--e1_intercept_aic_margin',
        type=float,
        default=2.0,
        help='When e1_intercept_mode=aic, prefer the best intercept model whenever its AIC is within this margin of the best no-intercept model. AIC differences below 2 are treated as practically indistinguishable.'
    )
    parser.add_argument(
        '--e2_from_e1_mask',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='E2 mode: read fixed f/c masks and basis names from E1 support_mask.npz instead of generating a new primary mask.'
    )
    parser.add_argument(
        '--e2_input_root',
        type=str,
        default='SIGN-phase/output/e1',
        help='Root directory containing E1 run artifacts.'
    )
    parser.add_argument(
        '--e2_output_root',
        type=str,
        default='SIGN-phase/output/e2',
        help='Root directory where E2 run artifacts are saved.'
    )
    parser.add_argument('--e2_run_name', type=str, default='', help='Optional E2 run directory name.')
    parser.add_argument(
        '--e2_init_from_phase1',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Initialize SIGN coefficient multipliers so the raw coefficients start at the E1 final coefficients.'
    )
    parser.add_argument('--e2_prune_tau', type=float, default=1e-3, help='RMS-normalized contribution threshold for E2 pruning.')
    parser.add_argument('--e2_prune_abs_threshold', type=float, default=5e-3, help='Absolute Phase-II coefficient threshold. Terms below this are removed.')
    parser.add_argument('--e2_prune_relative_threshold', type=float, default=1.5e-2, help='Phase-II per-channel relative coefficient threshold multiplied by the channel maximum.')
    parser.add_argument('--e2_decay_factor', type=float, default=0.9, help='Multiplicative decay applied each epoch to coefficients inside the Phase-II small-coefficient channel.')
    parser.add_argument('--e2_decay_abs_threshold', type=float, default=5e-3, help='Absolute floor for the Phase-II small-coefficient decay channel.')
    parser.add_argument('--e2_decay_relative_threshold', type=float, default=1.5e-2, help='Relative floor for the Phase-II small-coefficient decay channel.')
    parser.add_argument('--e2_decay_enabled', action=argparse.BooleanOptionalAction, default=True, help='Enable aggressive small-coefficient decay in Phase-II.')
    parser.add_argument('--e2_decay_during_warmup_only', action=argparse.BooleanOptionalAction, default=True, help='Restrict soft coefficient-mask decay to the sparse warmup period.')
    parser.add_argument('--e2_normalize_sparsity_loss', action=argparse.BooleanOptionalAction, default=True, help='Normalize coefficient L1 penalties by the active mask size.')
    parser.add_argument('--e2_warmup_lam_f', type=float, default=1e-5, help='Warmup penalty weight for function coefficients.')
    parser.add_argument('--e2_warmup_lam_c', type=float, default=1e-5, help='Warmup penalty weight for coupling coefficients.')
    parser.add_argument('--e2_reset_optimizer_after_warmup', action=argparse.BooleanOptionalAction, default=True, help='Reset Adam at the sparse-to-MSE objective switch.')
    parser.add_argument('--e2_post_warmup_lr', type=float, default=1e-3, help='Learning rate used after the sparse warmup; 0 reuses --lr.')
    parser.add_argument(
        '--e2_input_mask_threshold',
        type=float,
        default=1e-4,
        help='Threshold used when reading nonzero terms from an E1 support_mask.npz for Phase-II. Set to 0 for no extra absolute input-mask filtering.'
    )
    parser.add_argument('--e2_max_train_time_points', type=int, default=0, help='Optional cap on train time points for E2 speed; 0 keeps condition default.')
    parser.add_argument('--e2_min_epochs', type=int, default=30, help='Minimum Phase-II epochs before early stopping.')
    parser.add_argument('--e2_loss_patience', type=int, default=30, help='Stop when loss does not improve for this many epochs.')
    parser.add_argument('--e2_support_patience', type=int, default=40, help='Stop when pruned support size does not shrink for this many epochs and loss has plateaued.')
    parser.add_argument('--e2_loss_min_delta', type=float, default=1e-5, help='Minimum relative loss improvement counted by early stopping.')
    parser.add_argument('--e2_abs_loss_stop', type=float, default=0.0, help='Stop after e2_abs_loss_min_epochs when MSE is below this absolute value and support is stable; 0 disables.')
    parser.add_argument('--e2_abs_loss_min_epochs', type=int, default=8, help='Minimum epochs before the absolute-loss early stop may trigger.')
    parser.add_argument('--e2_max_walltime_sec', type=float, default=2700.0, help='Per-run wall-clock cap in seconds; 0 disables.')
    parser.add_argument('--e2_sparse_warmup_epochs', type=int, default=30, help='Initial Phase-II epochs using MSE plus coefficient sparsity losses.')


    parser.add_argument(
        "--decoder",
        type=str,
        default='DGSI',
        help="Type of decoder model (DGSI, CGSI).",
    )
    parser.add_argument(
        "--adjoint",
        type=bool,
        default='False',
        help="Type of decoder model (DGSI, CGSI).",
    )

    parser.add_argument(
        "--agg",
        type=str,
        default='add',
        help="agg function: add/mean.",
    )

    parser.add_argument(
        "--UseEdgeAttr",
        type=bool,
        default=False,
        help='use edge A'
    )

    parser.add_argument(
        "--UseLasso",
        type=bool,
        default=True,
        help='use Lasso'
    )


    parser.add_argument(
        "--dont_use_encoder",
        action="store_true",

        help="If true, replace encoder with distribution to be estimated",
        default=True,
    )
    parser.add_argument(
        "--lr_z",
        type=float,
        default=0.1,
        help="Learning rate for distribution estimation.",
    )


    parser.add_argument(
        "--save_folder",
        type=str,
        default="logs",
        help="Where to save the trained model, leave empty to not save anything.",
    )
    parser.add_argument(
        "--expername",
        type=str,
        default="",
        help="If given, creates a symlinked directory by this name in logdir"
        "linked to the results file in save_folder"
        "(be careful, this can overwrite previous results)",
    )
    parser.add_argument(
        "--sym_save_folder",
        type=str,
        default="../logs",
        help="Name of directory where symlinked named experiment is created."
    )
    parser.add_argument(
        "--load_folder",
        type=str,
        default='',

        help="Where to load pre-trained model if finetuning/evaluating. "
        + "Leave empty to train from scratch",
    )


    parser.add_argument(
        "--no_validate", action="store_true", default=False, help="Do not validate results throughout training."
    )
    parser.add_argument(
        "--no_cuda", action="store_true", default=False, help="Disables CUDA training."
    )
    parser.add_argument("--var", type=float, default=1e-3, help="Output variance.")

    parser.add_argument(
        "--invariant",
        type=bool,
        default=True,
        help="Use invariant data.",
    )


    args = parser.parse_args()
    args.test = True

    defaults = {
        'Kuramoto': {'network': 'power_law', 'dims': 1, 'init_scale': 10.0},
        'SIS': {'network': 'small_world', 'dims': 1, 'init_scale': 0.5},
        'Gene': {'network': 'small_world', 'dims': 1, 'init_scale': 1.0},
        'MM': {'network': 'small_world', 'dims': 1, 'init_scale': 1.0},
        'FHN': {'network': 'small_world', 'dims': 2, 'init_scale': 0.5},
        'HR': {'network': 'small_world', 'dims': 3, 'init_scale': 1.0},
        'Rossler': {'network': 'small_world', 'dims': 3, 'init_scale': 1.0},
    }
    model_defaults = defaults.get(args.ode_model, {'network': 'small_world', 'dims': 3, 'init_scale': 1.0})
    if args.network is None:
        args.network = model_defaults['network']
    if args.dims is None:
        args.dims = model_defaults['dims']
    if args.init_scale is None:
        args.init_scale = model_defaults['init_scale']
    if args.e1_condition == 'sparse200' and args.train_time_points <= 0:
        args.train_time_points = 200
    if args.e1_library == 'L1':
        args.poly_p, args.poly_n, args.activate = 7, 2, True
    elif args.e1_library == 'L2':
        # v3 removes six coupled trig terms and adds self exponentials.
        # One extra polynomial order keeps the raw pool large enough to
        # fill the exact 100-term mask for the 1D Kuramoto case.
        args.poly_p, args.poly_n, args.activate = (19, 3, True) if args.e1_basis_variant == 'trig_exp_v2' else (18, 3, True)
    elif args.e1_library == 'L3':
        args.poly_p, args.poly_n, args.activate = (31, 5, True) if args.e1_basis_variant == 'trig_exp_v2' else (30, 5, True)


    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.validate = not args.no_validate
    args.use_encoder = not args.dont_use_encoder
    args.time = datetime.datetime.now().isoformat().replace(':','')
    args.root = str(
        Path(args.data_root).expanduser()
        / '{}_{}_{}_{}_{}'.format(
            args.ode_model, args.num_atoms, args.network,
            args.time_interval, args.time_stamp,
        )
    )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device.type != "cpu":
        if args.GPU_to_use is not None:
            torch.cuda.set_device(args.GPU_to_use)
        torch.cuda.manual_seed(args.seed)
        args.num_GPU = 1
        args.batch_size_multiGPU = args.batch_size * args.num_GPU
    else:
        args.num_GPU = None
        args.batch_size_multiGPU = args.batch_size

    return args
