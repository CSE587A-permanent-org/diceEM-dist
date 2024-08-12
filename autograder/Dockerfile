# You can change these variables to use a different base image, but
# you must ensure that your base image inherits from one of ours.
# You can also override these at build time with --build-arg flags
ARG BASE_REPO=gradescope/autograder-base
ARG TAG=latest

FROM ${BASE_REPO}:${TAG}


ADD autograder/source /autograder/source

# This is where the results of the tests are stored
RUN mkdir /autograder/results

ADD diceEM/test_assignment.py /autograder/source/

RUN cp /autograder/source/run_autograder /autograder/run_autograder

# Ensure that scripts are Unix-friendly and executable
RUN dos2unix /autograder/run_autograder
RUN chmod +x /autograder/run_autograder

# Do whatever setup was needed in setup.sh, including installing apt packages
# Cleans up the apt cache afterwards in the same step to keep the image small
RUN apt-get update \ 
    && apt-get install -y build-essential --no-install-recommends make \
        ca-certificates \
        git \
        libssl-dev \
        zlib1g-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        wget \
        curl \
        llvm \
        libncurses5-dev \
        xz-utils \
        tk-dev \
        libxml2-dev \
        libxmlsec1-dev \
        libffi-dev \
        liblzma-dev \
        tzdata \
        && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Set the timezone to UTC
ENV TZ=UTC

# Set the system time in the container
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# symlink python3 to python
RUN ln -s /usr/bin/python3 /usr/bin/python

USER $USER
ARG HOME="/home/$USER"

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3 -

# Add Poetry to the PATH
ENV PATH="$HOME/.local/bin:$PATH"

# Set the poetry config for virtual environments
RUN poetry config virtualenvs.in-project true

# The base image defines the CMD and ENTRYPOINT, so don't redefine those