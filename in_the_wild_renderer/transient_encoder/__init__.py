
import importlib
from in_the_wild_renderer.transient_encoder.te_base import TEBase

def findTEModel(model_name: str):
    class_name = "in_the_wild_renderer.transient_encoder.te_" + model_name
    model_lib = importlib.import_module(class_name)
    model = None
    target_model_name = ('te' + model_name).replace('_', '')
    for name, cls in model_lib.__dict__.items():
        if name.lower() == target_model_name.lower()  and issubclass(cls, TEBase):
            model = cls
    if model is None:
        print('No such Transient Encoder Model, check the options for TE enc_type')
        exit(1)
    return model

def createTransientEncoder(opt,scene):
    te_model = findTEModel(opt.te_enc_type)
    instance = te_model(opt,scene)
    return instance



if __name__=="__main__":
    from arguments import TEParams
    from argparse import ArgumentParser, Namespace
    import sys

    parser = ArgumentParser(description="Test script parameters")
    tp = TEParams(parser)
    args = parser.parse_args(sys.argv[1:])
    app_encoder_opt = tp.extract(args)
    te = createTransientEncoder(app_encoder_opt)