import importlib
import collections


def load_from(path_spec: str):
    p, m = path_spec.rsplit('.', 1)
    mod = importlib.import_module(p)
    obj = getattr(mod, m)
    return obj


def _flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.Mapping):
            items.extend(_flatten_dict(v, None, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)