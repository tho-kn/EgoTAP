from .dataset_options import DatasetOptions

class FileCheckOptions(DatasetOptions):
    def initialize(self):
        DatasetOptions.initialize(self)
        self.parser.add_argument('--check_integrity', action='store_true')
        self.parser.add_argument('--check_json', action='store_true')
        self.parser.add_argument('--check_depth_image', action='store_true')
        self.parser.add_argument('--check_rgb_image', action='store_true')
