FROM continuumio/anaconda
RUN ["conda","install","-y","scipy"]
RUN ["pip","install","cvxopt"]

RUN mkdir -p /root/.config/matplotlib
RUN echo "backend : Agg" > /root/.config/matplotlib/matplotlibrc
