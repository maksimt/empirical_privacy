# empirical_privacy

### Basic how to get started

1. Clone & Build+Run Docker (the build takes 5-10 min):
```bash
git clone https://github.com/maksimt/empirical_privacy
cd empirical_privacy
docker build -t derivedjupyter .
docker run -p 8888:8888 -p 8082:8082 -v $(pwd):/emp_priv derivedjupyter:latest
```
2. Navigate to the jupyter-notebook running inside the docker container.
3. Open Notebooks/1-bit sum.ipynb and run the cells in order from top to bottom.

### Empirical Privacy Results on real datasets

Our results are available in the [Analysis Notebook](https://github.com/maksimt/empirical_privacy/blob/master/Notebooks/Analyze%20Completed%20CCCs.ipynb).

1. We ran the k-nearest neighbors classifier on the Movielens-1M recommender-systems dataset,
and the 20 newsgroups text dataset. The datasets are downloaded when the docker container is built
and the loading/pre-processing scripts we use are included in [dataset_utils](https://github.com/maksimt/empirical_privacy/tree/master/src/dataset_utils).
2. For each dataset we repeat for multiple trials:
    1. For each document that we care about:
        1. Generate `m/2` samples of statistics when the document is included in the dataset used for learning.
        2. Generate `m/2` samples of statistics when the document is not included in the dataset used for learning.
        3. Train a kNN classifier on the `m` samples, and record its accuracy on a held-out validation set.
3. The six statistics used by the k-nearest neighbors classifier are defined in [empirical_privacy.row_distributed_svd.gen_sample](src/empirical_privacy/row_distributed_svd.py).
They're based on the idea of weighing the correlation matrix `X^T X` by the out product of the document under investigation `x x^T`.

We also vary the dataset size, i.e. the number of binomial trials in the one-bit-sum case or the size of the
dataset used for learning as a fraction of the original 20NG or ML-1M dataset (we refer to this as the `part_fraction`).

**To reproduce** our results:
1. Open a terminal from inside Jupyter in the docker container.
2. Run `luigid`
3. Run the first two cells of the "Row Distributed SVD" notebook.
4. Run all the cells in the "Analyze Completed CCCs" notebook.

### Using the luigi-based sampling framework for your own empirical privacy experiments

[`luigi`](https://luigi.readthedocs.io/en/stable/index.html) is a python-based dependency specification framework.
It provides a central scheduler which makes it easy to parallelize the execution of a computation graph
while ensuring that work isn't duplicated and hardware is fully utilized.

We provide a framework that will orchestrate the experiments needed to measure empirical_privacy.
The goal is to minimize the amount of code that needs to be written for a new problem setting,
as well as take care of the implementation and testing for the key algorithms.

1. The main task is to implement a GenSample subclass that overrides the `gen_sample(sample_number)` method.
See the [one-bit-sum](src/empirical_privacy/one_bit_sum.py) example to start out, and then see [row_distributed_svd](src/empirical_privacy/row_distributed_svd.py).
Problem-specific parameters can be passed in the `dataset_settings` parameter.
2. Once that's done you can use the [build_convergence_curve_helper](src/luigi_utils/helpers.py) to
build a end-to-end pipeline with sensible defaults, or you can customize it by overriding the classes in [the framework](src/luigi_utils/sampling_framework.py).
3. To compute the targets using luigi they must be passed to `luigi.build` (see Notebooks for examples).
These will typically need to communicate to a luigi scheduler server, which you can run by opening a terminal from Juptyer and running `luigid`.
The scheduler will show you the progress of your computation on **localhost:8082**.

You may also be interested in my notes on [integrating with PyCharm](docs/Pycharm%20Integration.md).