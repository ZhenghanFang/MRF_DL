def getModel(opt):
	if opt.model == 'SimpleModel':
	    from models.simple_model import SimpleModel as Model
	elif opt.model == 'MultilossModel':
	    from models.multiloss_model import MultilossModel as Model
	elif opt.model == 'RegreClassModel':
	    from models.regre_class_model import RegreClassModel as Model
	elif opt.model == 'ResidueModel':
	    from models.residue_model import ResidueModel as Model
	return Model