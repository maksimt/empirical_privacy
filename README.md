# empirical_privacy

This repository contains the code necessary to reproduce results in our paper
*Empirical Methods for Estimating Privacy*.

* We use `docker` to create a reproducible execution environment.
* We use [`luigi`](https://luigi.readthedocs.io/en/stable/index.html) to express our privacy estimation algorithms as a DAG of dependencies. We also use `luigi` to define our experiments. The advantages of expressing experiments and the privacy estimation algorithms as a DAG is that the `luigi` scheduler can avoid re-computing previouslly computed results, and it can compute independent nodes in parallel.
* Each experiment described in our paper has a corresponding jupyter [notebook](https://github.com/maksimt/empirical_privacy/tree/master/Notebooks). Each experiment can be entirely reproduced by selecting "Kernel > Restart & Run All"

We hope that our privacy estimation algorithms, and experiment framework can be used to study 'privacy problem settings' that we haven't thought of. :)

### Requirements to run:
1. [`docker`](https://www.docker.com/products/docker-desktop).

### Basic how to get started

1. Clone & Build+Run Docker (the build takes 5-10 min):
```bash
git clone https://github.com/maksimt/empirical_privacy
cd empirical_privacy
docker build -t derivedjupyter .
docker run -p 8888:8888 -p 8082:8082 -v $(pwd):/emp_priv derivedjupyter:latest
```
2. Navigate to the jupyter-notebook running inside the docker container.
  1. Get the jupyter token `docker logs 2>&1 $(docker ps 2>&1 | grep derivedjupyter | awk '{print $1}') | grep token`
  2. Navigate to `127.0.0.1:8888` and enter the token you just got.
3. Open Notebooks/Experiment 1 -- Bootstrap Validation.ipynb.ipynb and run the cells in order from top to bottom.


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
