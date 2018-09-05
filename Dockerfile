FROM jupyter/scipy-notebook:177037d09156

MAINTAINER Maksim Tsikhanovich <github.com/maksimt>

VOLUME /emp_priv

WORKDIR /emp_priv

ENV PYTHONPATH "/emp_priv/src/:${PYTHONPATH}"

COPY requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt

USER root

RUN mkdir /datasets && \
    cd /datasets && \
    echo "Downloading Online News & Million Songs & MovieLens-1M Datasets" && \
    wget -q -t 3 -O OnlineNews.zip https://archive.ics.uci.edu/ml/machine-learning-databases/00332/OnlineNewsPopularity.zip && \
    wget -q -t 3 -O MillionSongs.zip https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip && \
    wget -q -t 3 -O ML1M.zip http://files.grouplens.org/datasets/movielens/ml-1m.zip && \
    unzip OnlineNews.zip && \
    unzip MillionSongs.zip && \
    unzip ML1M.zip && \
    rm ML1M.zip && \
    rm OnlineNews.zip && \
    rm MillionSongs.zip && \
    chown -R jovyan:users /datasets

RUN python -c "from sklearn.datasets.twenty_newsgroups import fetch_20newsgroups;\
                fetch_20newsgroups(data_home='/datasets', subset='all')"