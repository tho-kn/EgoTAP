from .dataset_options import DatasetOptions

class TestOptions(DatasetOptions):
    def initialize(self):
        DatasetOptions.initialize(self)

        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--show_heatmaps', action='store_true', help='show heatmaps')

        self.isTrain = False
        
    def parse(self):
        DatasetOptions.parse(self)
        self.opt.use_amp = False
        
        return self.opt
        