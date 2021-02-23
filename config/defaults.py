import multiprocessing

from yacs.config import CfgNode as CN

_C = CN()

_C.output_dir = 'output'
_C.trainer = 'Trainer'
_C.log_name = ""

# Benchmark different cudnn algorithms.
# If input images have very different sizes, this option will have large overhead
# for about 10k iterations. It usually hurts total time, but can benefit for certain models.
# If input images have the same or similar sizes, benchmark is often helpful.
_C.cudnn_benchmark = False
_C.device = 'cuda'
_C.num_gpus = 1
_C.upscale_factor = 2
_C.start_iter = 1
_C.random_seed = 0
_C.eval_only = False

_C.model = CN()
_C.model.name = 'EDSR'
_C.model.weights = ''
_C.model.pixel_mean = [103.530, 116.280, 123.675]
_C.model.in_channels = 3
_C.model.out_channels = 3

_C.dataset = CN()
_C.dataset.path = 'dataset'
_C.dataset.name = 'PreLoadDataset'
_C.dataset.train = ''
_C.dataset.test = ''
_C.dataset.interpolation = 'bicubic'
_C.dataset.input_size = 32
_C.dataset.gray = False
_C.dataset.re_generate = False
_C.dataset.pre_load = True
_C.dataset.cubic_a = -0.5

_C.dataloader = CN()
_C.dataloader.num_workers = min(multiprocessing.cpu_count(), 20)
_C.dataloader.batch_size = 32

# DPDNN settings
_C.dpdnn = CN()
_C.dpdnn.iteration = 6

_C.dbpn = CN()
_C.dbpn.num_features = 64
_C.dbpn.num_blocks = 7
_C.dbpn.norm_type = False
_C.dbpn.active = 'prelu'

_C.edsr = CN()
_C.edsr.num_features = 256
_C.edsr.num_blocks = 32
_C.edsr.res_scale = 0.1

_C.srfbn = CN()
_C.srfbn.num_features = 64
_C.srfbn.num_steps = 4
_C.srfbn.num_groups = 6
_C.srfbn.active = 'prelu'
_C.srfbn.norm_type = False

_C.rdn = CN()
_C.rdn.num_features = 64
_C.rdn.num_blocks = 16
_C.rdn.num_layers = 8

_C.meta = CN()
_C.meta.backbone = 'RDN'

_C.rcan = CN()
_C.rcan.n_resgroups = 10
_C.rcan.n_resblocks = 20
_C.rcan.n_feats = 64
_C.rcan.reduction = 16

_C.mirnet = CN()
_C.mirnet.num_features = 64
_C.mirnet.num_blocks = 2
_C.mirnet.num_groups = 3

_C.cyclesr = CN()
_C.cyclesr.srnet = 'RDN'
_C.cyclesr.synthetic_percent = 0.5
_C.cyclesr.loss_weights = (1, 1, 1e-4)  # pixel loss, consistency loss and discriminate loss
_C.cyclesr.train_real = []
_C.cyclesr.test_real = []
_C.cyclesr.gaussian_sigma = (5, 9, 15)
_C.cyclesr.gaussian_weights = (0.25, 0.5, 1.0)

_C.panet = CN()
_C.panet.num_resblocks = 20
_C.panet.num_features = 64
_C.panet.res_scale = 1

_C.optimizer = CN()
_C.optimizer.name = 'Adam'

_C.lr_scheduler = CN()
_C.lr_scheduler.name = 'MultiStepLR'
_C.lr_scheduler.start_lr = 0.1
_C.lr_scheduler.base_lr = 1e-4
_C.lr_scheduler.end_lr = 0.01
_C.lr_scheduler.warm_up_end_iter = 0.05
_C.lr_scheduler.cosine_start_iter = 0.8

_C.solver = CN()
_C.solver.max_iter = 30000
_C.solver.save_interval = 1000
_C.solver.test_interval = 500
_C.solver.loss = 'l1_loss'
_C.solver.multi_scale = ()
_C.solver.chop_size = 0

_C.tensorboard = CN()
_C.tensorboard.clear_before = True
_C.tensorboard.save_freq = 100
_C.tensorboard.image_num = 10
_C.tensorboard.name = ''
