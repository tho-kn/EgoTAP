
def create_model(opt):
    print(opt.model)

    if opt.model == "heatmap_shared":
        from .heatmap_shared_model import HeatmapSharedModel
        model = HeatmapSharedModel()

    elif opt.model == 'egotap_autoencoder':
        from .egotap_autoencoder_model import EgoTAPAutoEncoderModel
        model = EgoTAPAutoEncoderModel()

    else:
        raise ValueError('Model [%s] not recognized.' % opt.model)

    model.initialize(opt)
    print("model [%s] was created." % (model.name()))
    return model