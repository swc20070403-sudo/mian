import pickle, numpy as np
class SafeUnpickler(pickle.Unpickler):
    ALLOWED = {('numpy','dtype'), ('numpy.core.numeric','_frombuffer'), ('numpy._core.numeric','_frombuffer')}
    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            if name == 'dtype': return np.dtype
            from numpy._core import numeric
            return numeric._frombuffer
        raise pickle.UnpicklingError(f'blocked {module}.{name}')
def load(path):
    with open(path,'rb') as f:
        return SafeUnpickler(f).load()
