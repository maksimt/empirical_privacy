import itertools
from abc import abstractmethod, ABC
from collections import namedtuple
import six

import dill
import luigi
import numpy as np

from empirical_privacy.config import LUIGI_COMPLETED_TARGETS_DIR, \
    MIN_SAMPLES, SAMPLES_BASE
from experiment_framework.luigi_target_mixins import AutoLocalOutputMixin, \
    LoadInputDictMixin, DeleteDepsRecursively
from experiment_framework.python_helpers import load_from
from experiment_framework.calculations import accuracy_to_statistical_distance


def ComputeConvergenceCurve(
        compute_stat_dist: 'EvaluateStatisticalDistance') \
        -> '_ComputeConvergenceCurve':
    class T(_ComputeConvergenceCurve):
        pass

    T.compute_stat_dist = compute_stat_dist
    return T


_CP = namedtuple('CurvePoint', ['trial', 'training_set_size'])


class _ComputeConvergenceCurve(
    AutoLocalOutputMixin(base_path=LUIGI_COMPLETED_TARGETS_DIR),
    LoadInputDictMixin,
    DeleteDepsRecursively,
    luigi.Task,
    ABC
    ):
    n_trials_per_training_set_size = luigi.IntParameter()
    n_max = luigi.IntParameter()

    dataset_settings = luigi.DictParameter()
    validation_set_size = luigi.IntParameter(default=200)
    in_memory = luigi.BoolParameter(default=False)

    @property
    def priority(self):
        """
        The priority of a particular CCC. Higher is better.
        1. Lower n_max have higher priority
        2. Lower n_trials have higher priority
        """
        return 100*1/self.n_max + 1*1/self.n_trials_per_training_set_size

    @property
    def pow_min(self):
        return np.floor(np.log(MIN_SAMPLES) / np.log(SAMPLES_BASE)
                        + np.spacing(1)).astype(np.int)

    @property
    def n_steps(self):
        pow_max = np.floor(np.log(self.n_max) / np.log(SAMPLES_BASE)
                           + np.spacing(1)).astype(np.int)
        n_steps = pow_max - self.pow_min + 1
        assert n_steps > 0 , 'n_steps is {} which results in no ' \
                             'samples.'.format(n_steps)
        return n_steps

    @property
    def _training_set_sizes(self):
        return np.logspace(start=self.pow_min,
                           stop=self.pow_min + self.n_steps - 1,
                           num=self.n_steps,
                           dtype=int, base=SAMPLES_BASE)

    def requires(self):
        reqs = {}
        for training_set_size, trial in itertools.product(
                self._training_set_sizes,
                range(self.n_trials_per_training_set_size)
                ):
            reqs[_CP(trial, training_set_size)] = \
                self.compute_stat_dist(
                    dataset_settings=self.dataset_settings,
                    training_set_size=training_set_size,
                    validation_set_size=self.validation_set_size,
                    random_seed='trial{}'.format(trial),
                    in_memory=self.in_memory
                    )
        self.reqs_ = reqs
        return reqs

    def run(self):
        _inputs = self.load_input_dict()

        tss = self._training_set_sizes
        accuracy_matrix = np.empty(
            (self.n_trials_per_training_set_size, self.n_steps))

        for training_set_size, trial in itertools.product(
                tss,
                range(self.n_trials_per_training_set_size)
                ):
            col = np.argwhere(tss == training_set_size)[0, 0]
            accuracy_matrix[trial, col] = _inputs[_CP(trial, training_set_size)]['accuracy']

        self.output_ = {'accuracy_matrix': accuracy_matrix,
                       'training_set_sizes':tss}
        with self.output().open('wb') as f:
            dill.dump(self.output_, f, 2)


def EvaluateStatisticalDistance(samplegen: '_GenSamples',
                                model: '_FitModel') \
        -> '_EvaluateStatisticalDistance':
    class T(_EvaluateStatisticalDistance):
        pass

    T.samplegen = samplegen
    T.model = model
    return T


class _EvaluateStatisticalDistance(
    AutoLocalOutputMixin(base_path=LUIGI_COMPLETED_TARGETS_DIR),
    LoadInputDictMixin,
    DeleteDepsRecursively,
    luigi.Task,
    ABC
    ):
    dataset_settings = luigi.DictParameter()
    training_set_size = luigi.IntParameter()
    validation_set_size = luigi.IntParameter()
    random_seed = luigi.Parameter()

    in_memory = luigi.BoolParameter(default=False)

    def requires(self):
        reqs = {}
        reqs['model'] = self.model(
            dataset_settings=self.dataset_settings,
            samples_per_class=int(round(self.training_set_size / 2)),
            random_seed=self.random_seed,
            in_memory = self.in_memory
            )
        reqs['samples_positive'] = self.samplegen(
            dataset_settings=self.dataset_settings,
            num_samples=int(round(self.validation_set_size / 2)),
            random_seed='validation_seed{}_pos'.format(self.random_seed),
            generate_positive_samples=True,
            )
        reqs['samples_negative'] = self.samplegen(
            dataset_settings=self.dataset_settings,
            num_samples=int(round(self.validation_set_size / 2)),
            random_seed='validation_seed{}_neg'.format(self.random_seed),
            generate_positive_samples=False,
            )
        self.reqs_ = reqs
        if self.in_memory:
            return {}
        return reqs

    def run(self):
        _input = self.compute_or_load_requirements()
        accuracy = self.model.compute_classification_accuracy(
            _input['model'],
            _input['samples_positive'],
            _input['samples_negative']
            )
        sd = accuracy_to_statistical_distance(accuracy)
        self.output_ = {'statistical_distance': sd,
                       'accuracy': accuracy}
        with self.output().open('wb') as f:
            dill.dump(self.output_, f, 0)


def FitModel(gen_samples_type):
    class T(_FitModel):
        pass

    T.gen_samples_type = gen_samples_type
    T.model = None
    return T


class _FitModel(AutoLocalOutputMixin(base_path=LUIGI_COMPLETED_TARGETS_DIR),
                LoadInputDictMixin,
                DeleteDepsRecursively,
                luigi.Task,
                ABC):
    dataset_settings = luigi.DictParameter()
    samples_per_class = luigi.IntParameter()
    random_seed = luigi.Parameter()

    in_memory = luigi.BoolParameter(default=False)

    @abstractmethod
    def fit_model(self, negative_samples, positive_samples):
        """
        Given positive and negative samples return a fitted model
        Parameters
        """
        pass

    @classmethod
    def compute_classification_accuracy(cls, model, *samples):
        """
        Parameters
        ----------
        model : dict
            All the data that represents a fitted model
        samples : list of {'X':array, 'y':array}
            The samples on which we should compute statistical distance
        Returns
        -------
        float : the statistical distance

        """
        raise NotImplementedError()

    def requires(self):
        req = {}
        req['samples_positive'] = self.gen_samples_type(
            dataset_settings=self.dataset_settings,
            num_samples=self.samples_per_class,
            random_seed=self.random_seed,
            generate_positive_samples=True,
            )
        req['samples_negative'] = self.gen_samples_type(
            dataset_settings=self.dataset_settings,
            num_samples=self.samples_per_class,
            random_seed=self.random_seed,
            generate_positive_samples=False,
            )
        self.reqs_ = req
        if not self.in_memory:
            return req
        return {}

    def run(self):
        _input = self.compute_or_load_requirements()

        # We set the random seed since the model's fitter may use
        # numbers from the random stream.
        GenSample.set_simple_random_seed(sample_number=-1,
                                         random_seed=self.random_seed)
        model = self.fit_model(_input['samples_negative'],
                               _input['samples_positive'])
        self.output_ = model
        with self.output().open('wb') as f:
            dill.dump(model, f, 2)


class _GenSamples(
    AutoLocalOutputMixin(base_path=LUIGI_COMPLETED_TARGETS_DIR),
    LoadInputDictMixin,
    DeleteDepsRecursively,
    luigi.Task,
    ABC
    ):
    dataset_settings = luigi.DictParameter()
    random_seed = luigi.Parameter()
    generate_positive_samples = luigi.BoolParameter()
    num_samples = luigi.IntParameter()

    def requires(self):
        if not self.generate_in_batch:
            GS = self.gen_sample_type
            reqs = [GS(
                dataset_settings=self.dataset_settings,
                random_seed=self.random_seed,
                generate_positive_sample=self.generate_positive_samples,
                sample_number=sample_num

                )
                for sample_num in range(self.num_samples)]
            reqs = {'samples': reqs}
        elif self.generate_in_batch and self.num_samples > MIN_SAMPLES:
            self.n_prev = np.floor(self.num_samples / SAMPLES_BASE).astype(int)
            reqs = {
                'prev': self.__class__(  # because the class will have
                    # been specialized by the factory GenSamples()
                    dataset_settings=self.dataset_settings,
                    random_seed=self.random_seed,
                    generate_positive_samples=self.generate_positive_samples,
                    num_samples=self.n_prev
                    )
                }
        else:
            self.n_prev = 0
            reqs = {}
        self.reqs_ = reqs

        if self.in_memory:
            return {}
        return reqs

    def run(self):
        prev = {'X': None, 'y': None}
        if not self.generate_in_batch:
            samples = self.compute_or_load_requirements()['samples']
        else:  # self.generate_in_batch
            if self.num_samples > MIN_SAMPLES:
                prev = self.compute_or_load_requirements()['prev']

            f_GS = self.gen_sample_type(dataset_settings=self.dataset_settings,
                                        random_seed=self.random_seed,
                                        generate_positive_sample=self.generate_positive_samples
                                        ).gen_sample

            samples = []
            for sn in range(self.n_prev, self.num_samples):
                # set_progress_percentage is a blocking network IO operation... lol
                # self.set_progress_percentage(100*(sn-self.n_prev)/n_to_make)
                samples.append(f_GS(sample_number=sn))
        X, y = zip(*samples)
        if prev['X'] is not None:
            X = self.x_concatenator((prev['X'], self.x_concatenator(X)))
        else:
            X = self.x_concatenator(X)
        if prev['y'] is not None:
            y = self.y_concatenator((prev['y'], self.y_concatenator(y)))
        else:
            y = self.y_concatenator(y)
        output = {'X': X, 'y': y}
        self.output_ = output

        with self.output().open('w') as f:
            dill.dump(output, f, 2)


Sample = namedtuple('Sample', ['x', 'y'])


class GenSample(
    AutoLocalOutputMixin(base_path=LUIGI_COMPLETED_TARGETS_DIR),
    LoadInputDictMixin,
    DeleteDepsRecursively,
    luigi.Task,
    ABC
    ):
    dataset_settings = luigi.DictParameter()
    random_seed = luigi.Parameter()
    generate_positive_sample = luigi.BoolParameter()
    sample_number = luigi.IntParameter(default=-1337)

    @classmethod
    def set_simple_random_seed(cls, sample_number, random_seed):
        seed_val = hash('{seed}sample{s_num}'.format(seed=random_seed,
                                                     s_num=sample_number))
        seed_val %= 4294967296
        np.random.seed(seed_val)

    @abstractmethod
    def gen_sample(self, sample_number: int):
        raise NotImplementedError('This method needs to be implemented by a '
                                  'subclass of GenSample.')
        # return x, y

    def run(self):
        if self.sample_number == -1337:
            raise ValueError('A sample_number wasnt provided to the '
                             'constructor of GenSample() but the run() method '
                             'was called.'
                             'The sample_number argument is only optional if '
                             'the gen_sample() method is called directly.')
        x, y = self.gen_sample(sample_number=self.sample_number)

        self.output_ = Sample(x, y)
        with self.output().open('wb') as f:
            dill.dump(self.output_, f, 2)


def GenSamples(gen_sample_type, x_concatenator=np.concatenate,
               y_concatenator=np.concatenate, generate_in_batch=False) -> _GenSamples:
    """
    Parameters
    ----------
    gen_sample_type : class
        The class that will be generating samples
    x_concatenator : function
        Takes tuples of the form (x1, x2) where xi are either elements or
        outputs of x_concatenator and concatenates them. E.g.
        np.conconcatenate, np.vstack or np.hstack
    y_concatenator : function
        Takes tuples of the form (y1, y2) where yi are either elements or
        outputs of x_concatenator and concatenates them.
    generate_in_batch : bool, optional (default False)
        Generate the entire batch of samples directly without spawning subtasks
        Can improve performance if the IO cost of saving/loading a sample is
        higher than computing it.
    Returns
    -------
    T : class
    """

    class T(_GenSamples):
        pass

    T.gen_sample_type = gen_sample_type
    for attr_name, attr in [('x_concatenator', x_concatenator),
                            ('y_concatenator', y_concatenator)]:
        if isinstance(attr, six.string_types):
            attr = load_from(attr)
        setattr(T, attr_name, staticmethod(attr))
    T.generate_in_batch = generate_in_batch
    T.in_memory = generate_in_batch
    return T