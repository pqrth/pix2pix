FROM kaixhin/cuda-torch

RUN luarocks install nngraph && luarocks install https://raw.githubusercontent.com/szym/display/master/display-scm-0.rockspec && apt install wget

COPY . /app
WORKDIR /app
