import collections
import pickle
import os

import luigi
import numpy as np

class LoadInputDictMixin(luigi.Task):
    """luigi Task wrapper that automatically generates the output() method"""


    def load_input_dict(self, _input = None, all_numpy = False):
        """load dict of inputs luigi Targets and return their loaded values
        as output. Supports nested dicts."""
        inp = {}
        if not _input:
            _input = self.input()

        if type(_input)==list:
            if len(_input)==1:
                _input = _input[0]
            else:
                raise Exception('load_input_dict: unexpected input format '
                                '{0}'.format(type(_input)))
        for k in _input:
            if type(_input[k]) == dict:
                inp[k] = self.load_input_dict(_input[k], all_numpy=all_numpy)
            else:
                with _input[k].open() as f:
                    if not all_numpy:
                        inp[k] = pickle.load(f)
                    else:
                        inp[k] = np.load(f)
        return inp

    def load_completed_reqs(self):
        paths = []
        fns = os.listdir(self.base_path)
        if hasattr(self.input(), '__iter__'):
            for inp in self.input():
                fn = inp.path.split('/')[-1]
                if fn in fns:
                    paths.append(fn)
        elif hasattr(self.input(), 'path'):
            paths.append(self.input().path)
        else:
            paths.append(self.input())
        return paths

def _try_delete(path):
    """
    Attempt to delete file at path, pass on file not found, raise anything else

    Parameters
    ----------
    path: full or relative path to file

    Returns
    -------

    """

    try:
        os.remove(path)
    except OSError as err:  # file does not exist
        if 'File does not exist' in str(err):
            pass
        else:
            raise err

class SingleFileMTask(object):
    def output(self):
        return luigi.LocalTarget(gen_fn_v3(self.base_path, self))

    def delete_outputs(self):
        _try_delete(self.output().path)


class DictMTask(object):
    def output(self):
        dict_out = {}
        for k in self.outputs:
            dict_out[k] = luigi.LocalTarget(
                gen_fn_v3(self.base_path, self, suffix=repr(k))
            )
        return dict_out

    def delete_outputs(self):
        od = self.output()
        for k in od:
            _try_delete(od[k].path)


class ListMTask(object):
    def output(self):
        list_out = []
        for v in self.outputs:
            list_out.append(luigi.LocalTarget(
                gen_fn_v3(self.base_path, self)
            ))
        return list_out

    def delete_outputs(self):
        for it in self.output():
            _try_delete(it.path)


def AutoLocalOutputMixin(output='single_file', base_path='/tmp/'):
    if output == 'single_file':
        class T(SingleFileMTask):
            pass
        T.base_path = base_path
        return T
    elif isinstance(output, collections.Sequence):
        class T(ListMTask):
            pass
    elif isinstance(output, collections.Mapping):
        class T(DictMTask):
            pass
        T.outputs = output#luigi.DictParameter(default=output)
        T.base_path = base_path#luigi.Parameter(default=base_path)
        return T
    else:
        raise NotImplementedError('Output type {} not implemented'.format(
            type(output)))

def gen_fn_v3(base_path, obj, suffix=''):
    if not base_path.endswith('/'):
        base_path+='/'

    fn = base_path + repr(obj) + suffix

    if True or len(fn)>200:
        fn = base_path + obj.__class__.get_task_family() + '__' + \
             str(hash(obj)^hash(suffix))

    return fn

