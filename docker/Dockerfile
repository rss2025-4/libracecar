ARG BASE_IMAGE=staffmitrss/racecar-real:latest

FROM ${BASE_IMAGE}

USER root

# install docker
COPY ./docker/install_docker.sh /
RUN bash /install_docker.sh

# apt packages
RUN apt-get update \
    && apt-get install -y \
    direnv \
    python3.10-venv \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN deluser racecar || true
RUN groupdel zed || true
# Create a user
ARG DOCKER_UID=1000
ARG DOCKER_GID=1000
RUN groupadd -g "$DOCKER_GID" dockergroup \
    && useradd -u "$DOCKER_UID" -g "$DOCKER_GID" -ms /bin/bash dockeruser \
    && usermod -aG sudo dockeruser \
    && echo "dockeruser ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

USER dockeruser
WORKDIR /home/dockeruser

# install poetry
RUN curl -sSL https://install.python-poetry.org | python3 - \
    && ~/.local/bin/poetry self add poetry-plugin-shell \
    && ~/.local/bin/poetry self add poetry-plugin-up

# # create venv
# ENV VENV_PATH="/home/dockeruser/venv"
# COPY ./docker/create_venv.sh /
# COPY ./pyproject.toml /tmp_poetry/pyproject.toml
# COPY ./poetry.lock /tmp_poetry/poetry.lock
# RUN bash /create_venv.sh

# finalize, fast stuff
USER root
# RUN chmod -R a+rw /root
RUN chown -R dockeruser /root

COPY ./docker/bash.bashrc /etc/
COPY ./docker/default.rviz /default.rviz

# COPY ./entrypoint.sh /entrypoint.sh
# RUN chmod 0755 /entrypoint.sh
# ENTRYPOINT ["/entrypoint.sh"]

# caller must overwrite on docker run
ENTRYPOINT []

USER dockeruser
RUN mkdir -p /home/dockeruser/racecar_ws/src
RUN	echo 'eval "$(direnv hook bash)"' >> ~/.bashrc
RUN	echo 'export TERM=xterm-256color' >> ~/.bashrc
