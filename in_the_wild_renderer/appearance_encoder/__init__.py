
import importlib
from in_the_wild_renderer.appearance_encoder.ae_base import AEBase

def findAEModel(model_name: str):
    class_name = "in_the_wild_renderer.appearance_encoder.ae_" + model_name
    model_lib = importlib.import_module(class_name)
    model = None
    target_model_name = ('ae' + model_name).replace('_', '')
    for name, cls in model_lib.__dict__.items():
        if name.lower() == target_model_name.lower()  and issubclass(cls, AEBase):
            model = cls
    if model is None:
        print('No such Appearance Encoder Model, check the options for AE enc_type')
        exit(1)
    return model

def createAppearanceEncoder(opt,scene):
    ae_model = findAEModel(opt.enc_type)
    instance = ae_model(opt,scene)
    return instance



if __name__=="__main__":
    from arguments import AEParams
    from argparse import ArgumentParser, Namespace
    import sys

    parser = ArgumentParser(description="Test script parameters")
    ap = AEParams(parser)
    args = parser.parse_args(sys.argv[1:])
    app_encoder_opt = ap.extract(args)
    ae = createAppearanceEncoder(app_encoder_opt)