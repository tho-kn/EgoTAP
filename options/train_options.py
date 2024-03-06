from .dataset_options import DatasetOptions

class TrainOptions(DatasetOptions):
    def initialize(self):
        DatasetOptions.initialize(self)

        # ------------------------------ training epoch ------------------------------ #
        self.parser.add_argument('--epoch_count', type=int, default=1,
                                 help='the starting epoch count')
        self.parser.add_argument('--niter', type=int, default=0,
                                 help='# of iter with initial learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=0,
                                 help='# of iter to decay learning rate to zero')
        self.parser.add_argument('--continue_train', action='store_true',
                                 help='continue training: load the latest model')
        self.parser.add_argument('--transform_epoch', type=int, default=0,
                                 help='# of epoch for transform learning')
        self.parser.add_argument('--task_epoch', type=int, default=0,
                                 help='# of epoch for task learning')


        # ------------------------------ learning rate and loss weight ------------------------------ #
        self.parser.add_argument('--optimizer_type', type=str, default='Adam',
                                 help='optimizer type[Adam|SGD|DAdam|DSGD|DAdaGrad|Prodigy]')
        self.parser.add_argument('--lr_policy', type=str, default='lambda',
                                 help='learning rate policy[lambda|step|exponent|cos_anneal]')
        self.parser.add_argument('--lr_decay_iters_step', type=int, default=4,
                                 help='of iter to decay learning rate with a policy [step]')
        self.parser.add_argument('--lr', type=float, default=1e-3,
                                 help='initial learning rate for adam')
        self.parser.add_argument('--weight_decay', type=float, default=0.0,
                                 help='weight decay')
        self.parser.add_argument('--growth_rate', type=float, default=float('inf'),
                                 help='growth rate for DAdapt optimizer')
        self.parser.add_argument('--d_coef', type=float, default=1.0,
                                help='d_coef for DAdam optimizer')
        self.parser.add_argument('--opt_eps', type=float, default=1e-4,
                                help='eps for DAdam optimizer')
        self.parser.add_argument('--decouple', action='store_true',
                                 help='decoupled weight decay like AdamW')

        self.parser.add_argument('--lambda_mpjpe', type=float, default=1.0,
                                 help='weight for loss_mpjpe')
        self.parser.add_argument('--lambda_pelvis', type=float, default=0.01,
                                 help='weight for loss_pelvis')
        self.parser.add_argument('--lambda_rot', type=float, default=1.0,
                                 help='weight for loss_rot')
        self.parser.add_argument('--lambda_heatmap', type=float, default=1.0,
                                 help='weight for loss_heatmap')
        self.parser.add_argument('--lambda_segmentation', type=float, default=1.0,
                                 help='weight for loss_segmentation')
        self.parser.add_argument('--lambda_rot_heatmap', type=float, default=1.0,
                                 help='weight for loss_rot_heatmap')
        self.parser.add_argument('--lambda_pose', type=float, default=1e-1,
                                 help='weight for loss_pose')
        self.parser.add_argument('--lambda_indep_pos', type=float, default=1e-1,
                                 help='weight for loss_indep_pos')
        self.parser.add_argument('--lambda_heatmap_rec', type=float, default=1e-3,
                                 help='weight for loss_heatmap_rec')
        self.parser.add_argument('--lambda_rot_heatmap_rec', type=float, default=1e-3,
                                 help='weight for loss_rot_heatmap_rec')
        self.parser.add_argument('--lambda_cos_sim', type=float, default=-1e-2,
                                 help='weight for loss_cos_sim')

        # ------------------------------ display the results ------------------------------ #
        self.parser.add_argument('--display_freq', type=int, default=1,
                                 help='frequency of showing training results on screen')
        self.parser.add_argument('--print_epoch_freq', type=int, default=1,
                                 help='frequency of showing training results at the end of epochs')
        self.parser.add_argument('--save_latest_freq', type=int, default=1,
                                 help='frequency of saving the latest results')
        self.parser.add_argument('--val_epoch_freq', type=int, default=1,
                                 help='frequency of validation')
        self.parser.add_argument('--save_epoch_freq', type=int, default=1,
                                 help='frequency of saving checkpoints at the end of epochs')
        
        # ------------------------------ others ------------------------------ #
        self.parser.add_argument('--stage', action="append", dest="train_stage", default=[],
                                 help='train stage [rot|pos|hierarchy|final]')
        self.parser.add_argument('--auto_restart', action='store_true',
                                 help='auto restart training on failure')
        self.parser.add_argument('--auto_terminate', action='store_true',
                                 help='auto terminate training on failure')

        self.isTrain = True
