def getDataset(opt):
    if opt.dataset == 'mrf_dataset':
        from data.mrf_dataset import MRFDataset
    elif opt.dataset == 'single_dataset':
        from data.single_dataset import MRFDataset
    elif opt.dataset == 'motion_dataset':
        from data.motion_dataset import MRFDataset
    elif opt.dataset == 'residue_dataset':
        from data.residue_dataset import MRFDataset
    elif opt.dataset == 'highres_dataset':
        from data.highres_dataset import MRFDataset
    elif opt.dataset == 'threeD_dataset':
        from data.threeD_dataset import MRFDataset
    elif opt.dataset == 'simulated_dataset':
        from data.simulated_dataset import MRFDataset
    elif opt.dataset == 'dict_dataset':
        from data.dict_dataset import MRFDataset
    elif opt.dataset == 'single_dataset_2':
        from data.single_dataset_2 import MRFDataset
    elif opt.dataset == 'T1hT2_dataset':
        from data.T1hT2_dataset import MRFDataset
    elif opt.dataset == 'simulated_dataset_noise':
        from data.simulated_dataset_noise import MRFDataset
    elif opt.dataset == 'threeD_dataset_2':
        from data.threeD_dataset_2 import MRFDataset
    elif opt.dataset == 'threeD_dataset_3':
        from data.threeD_dataset_3 import MRFDataset
    elif opt.dataset == 'simulated_SVD_dataset':
        from data.simulated_SVD_dataset import MRFDataset
    else:
        raise ValueError('dataset type %s not recognized' % opt.dataset)
    return MRFDataset
