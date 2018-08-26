FROM jupyter/scipy-notebook:177037d09156

MAINTAINER Maksim Tsikhanovich <github.com/maksimt>

WORKDIR /emp_priv

ENV PYTHONPATH "/emp_priv/src/:${PYTHONPATH}"

COPY requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt

ENTRYPOINT ["/bin/sh", "-c"]