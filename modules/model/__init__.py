from .gcn_encoder import Graph_JP_estimator
from .uip import UIP


MODELS = {"GNN_JP":Graph_JP_estimator ,"UIP":UIP}

def get_model(args, parser):
    model_cls = MODELS[args.network]
    model_cls.add_args(parser)
    return model_cls