import inspect
import torch

def add_to_class(Class):
    return lambda obj: setattr(Class, obj.__name__, obj)

class NNBase(torch.nn.Module):
    def __init__(self):
        super().__init__()

""" 
Credit for function goes to:
https://www.d2l.ai/chapter_appendix-tools-for-deep-learning/utils.html
"""
@add_to_class(NNBase)
def save_hyperparameters(self, ignore=[]):
    frame = inspect.currentframe().f_back 
    _, _, _, local_vars = inspect.getargvalues(frame)
    self.hparams = {k:v for k, v in local_vars.items()
                    if k not in set(ignore+['self']) and not k.startswith('_')}
    self.__dict__.update(self.hparams)
    