# empirical_privacy

### Basic how to get started

1. Clone & Build+Run Docker
```bash
cd /tmp/
git clone https://github.com/maksimt/empirical_privacy
cd empirical_privacy
docker build -t derivedjupyter .
docker run -p 8888:8888 -v $(pwd):/emp_priv derivedjupyter:latest
```
2. Navigate to the jupyter-notebook running inside the docker container.
3. Open Notebooks/1-bit sum.ipynb and run the cells in order from top to bottom.

You may also be interested in my notes on [integrating with PyCharm](docs/Pycharm%20Integration.md).