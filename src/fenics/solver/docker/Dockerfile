FROM dolfinx/dolfinx:v0.7.3

COPY docker/requirements/base.txt /tmp/base.txt
RUN pip install -r /tmp/base.txt

RUN apt update && \
     apt install -y xvfb mesa-utils

# the dev container setup from MaG does not work with those libs editable installed in /root/
ARG PIP_FLAGS=-e

RUN git clone https://github.com/FEniCS/fiat.git
RUN cd ./fiat && git checkout tags/2019.1.0
RUN pip3 install $PIP_FLAGS ./fiat

RUN git clone https://github.com/colinRawlings/Irksome.git
RUN cd Irksome && git checkout tags/0.0.1
RUN pip3 install $PIP_FLAGS ./Irksome

RUN mkdir /work

# add pyright
RUN apt-get update \
  && apt-get install -yq curl

RUN curl https://raw.githubusercontent.com/creationix/nvm/master/install.sh | bash

# n.b. we are running non-interactively ...
ENV NVM_DIR="/root/.nvm"
RUN . "$NVM_DIR/nvm.sh" && nvm install 16
RUN . "$NVM_DIR/nvm.sh" && npm i -g pyright@1.1.335
RUN . "$NVM_DIR/nvm.sh" && pyright --version

ENTRYPOINT [ "/bin/bash", "-l", "-c" ]
