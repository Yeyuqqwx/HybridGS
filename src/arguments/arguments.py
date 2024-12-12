from argparse import Namespace


args = Namespace()
args.pipe = Namespace()
args.gs3d = Namespace()
args.dataset = Namespace()
args.opt = Namespace()


args.pipe.separate_sh = True
args.pipe.convert_SHs_python = False
args.pipe.compute_cov3D_python = False
args.pipe.debug = False


args.gs3d.sh_degree = 3

args.dataset.source_path = ""
args.dataset.images = "images"
args.dataset.resolution = -1
args.dataset.white_background = False
args.dataset.data_device = "cuda"
args.dataset.eval = False

args.opt.iterations = 30_000
args.opt.position_lr_init = 0.00016
args.opt.position_lr_final = 0.0000016
args.opt.position_lr_delay_mult = 0.01
args.opt.position_lr_max_steps = 30_000
args.opt.feature_lr = 0.0025
args.opt.opacity_lr = 0.025
args.opt.scaling_lr = 0.005
args.opt.rotation_lr = 0.001
args.opt.percent_dense = 0.01
args.opt.lambda_dssim = 0.2
args.opt.densification_interval = 100
args.opt.opacity_reset_interval = 3000
args.opt.densify_from_iter = 500
args.opt.densify_until_iter = 15_000
args.opt.densify_grad_threshold = 0.0002
args.opt.random_background = False
args.opt.optimizer_type = "sparse_adam"

############################### ++

args.description = ""
args.model_path = ""
args.only_run_for_render = False
args.opt.warming_up = 0

args.opt.save_iterations = [7000, 15000, 20000, 30000, 60000]