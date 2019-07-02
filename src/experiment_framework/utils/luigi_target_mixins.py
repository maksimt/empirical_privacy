import collections
import os

import dill
import luigi
import numpy as np
from luigi.task import task_id_str


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
        return 1
    except OSError as err:  # file does not exist
        if err.errno == 2:
            return 0
        else:
            raise err

class DeleteDepsRecursively(luigi.Task):
    def delete_deps(self):
        _input = self.deps()
        n_del = 0
        if type(_input) == dict:
            for k in _input:
                n_del += _input[k].delete_deps()
        elif type(_input) == list:
            for item in _input:
                n_del += item.delete_deps()
        else:
            if _input is not None:
                n_del += _try_delete(_input.path)
        n_del += _try_delete(self.output().path)
        return n_del


class LoadInputDictMixin(luigi.Task):
    """luigi Task wrapper that automatically generates the output() method"""

    def load_input_dict(self, _input=None, all_numpy=False):
        """load dict of inputs luigi Targets and return their loaded values
        as output. Supports nested dicts."""
        inp = {}
        if not _input:
            _input = self.input()

        if type(_input) == dict:
            for k in _input:
                inp[k] = self.load_input_dict(_input[k], all_numpy=all_numpy)
        elif type(_input) == list:
            inp = []
            for item in _input:
                inp.append(self.load_input_dict(item, all_numpy=all_numpy))
        else:
            with _input.open() as f:
                if not all_numpy:
                    inp = dill.load(f)
                else:
                    inp = np.load(f)
        return inp

    def compute_or_load_requirements(self):
        if not self.in_memory:
            _input = self.load_input_dict()
        else:
            _input = self.reqs_
            for key, obj in _input.items():
                _input[key] = self._populate_obj(obj)
        return _input

    @staticmethod
    def _populate_obj(obj):
        if hasattr(obj, 'complete') and obj.complete():
            with obj.output().open() as f:
                return dill.load(f)
        if hasattr(obj, 'requires'):
            obj.requires()  # to get the object to populate reqs_
            obj.run()
            return obj.output_
        elif type(obj) == list:
            for i in range(len(obj)):
                obj[i] = LoadInputDictMixin._populate_obj(obj[i])
            return obj

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


class SingleFileMTask(object):
    def output(self):
        return luigi.LocalTarget(
            gen_fn_v3(self.base_path, self),
            format=luigi.format.Nop
            )

    def delete_outputs(self):
        _try_delete(self.output().path)


class DictMTask(object):
    def output(self):
        dict_out = {}
        for k in self.outputs:
            dict_out[k] = luigi.LocalTarget(
                gen_fn_v3(self.base_path, self, suffix=repr(k)),
                format=luigi.format.Nop
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
                gen_fn_v3(self.base_path, self),
                format=luigi.format.Nop
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

        T.outputs = output  # luigi.DictParameter(default=output)
        T.base_path = base_path  # luigi.Parameter(default=base_path)
        return T
    else:
        raise NotImplementedError('Output type {} not implemented'.format(
            type(output)))


def gen_fn_v3(base_path, obj, suffix=''):
    fn = task_id_str(obj.task_family, obj.to_str_params())+suffix
    return os.path.join(base_path, fn)
