# Start from a Jupyter Docker Stacks version
FROM quay.io/jupyter/docker-stacks-foundation:python-3.12

# Needed for poetry package management: no venv, latest poetry, GRANT_SUDO don't work :(
ENV POETRY_VIRTUALENVS_CREATE=true \
    POETRY_VERSION=2.1.1 \
    GRANT_SUDO=yes

# The docker stacks make sudo very difficult, so we [just be root™]
USER root
RUN sudo apt update && \
    sudo apt upgrade -y && \
    sudo apt install -y curl autoconf automake libtool pkg-config && \
    rm -rf /var/lib/apt/lists/*

# Go back to jovyan user so we don't have permission problems
USER ${NB_USER}

# Install poetry so we can install our package requirements
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH "/home/jovyan/.local/bin:$PATH"

# Copy our poetry configuration files as jovyan user
COPY --chown=${NB_UID}:${NB_GID} pyproject.toml "/home/${NB_USER}/work/"
COPY --chown=${NB_UID}:${NB_GID} poetry.lock    "/home/${NB_USER}/work/"

# Install our package requirements via poetry. No venv. Squash max-workers error.
WORKDIR "/home/${NB_USER}/work"
RUN poetry config virtualenvs.create true && \
    poetry config installer.max-workers 10 && \
    poetry install --no-interaction --no-ansi --no-root -vvv && \
    poetry cache clear pypi --all -n
