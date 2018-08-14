FROM jupyter/scipy-notebook:177037d09156

MAINTAINER Maksim Tsikhanovich <github.com/maksimt>

WORKDIR /emp_priv

ADD . /emp_priv

RUN export PYTHONPATH="/emp_priv:$PYTHONPATH"
