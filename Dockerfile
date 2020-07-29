# Base image
FROM pytorch/pytorch:1.5.1-cuda10.1-cudnn7-runtime

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Non-root user with sudo access
ARG USERNAME=default
ARG USER_UID=1000
ARG USER_GID=$USER_UID

# Pin
ARG PIN_VERSION=3.13-98189-g60a6ef199
ENV PIN_HOME=/opt/pin-${PIN_VERSION}-gcc-linux

# Parsec
ARG PARSEC_VERSION=3.0
ENV PARSECDIR=/opt/parsec-${PARSEC_VERSION}
ENV PATH=${PATH}:${PARSECDIR}/bin
ENV MANPATH=${MANPATH}:${PARSECDIR}/man

# Java
ENV JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

# Configure apt
RUN apt-get update \
    && apt-get -y install --no-install-recommends apt-utils dialog 2>&1 \
    #
    # Install apt deps
    && apt-get -y install \
    sudo \
    git \
    build-essential \
    curl \
    m4 \
    openssh-client \
    openjdk-8-jdk \
    libssl-dev \
    libxext-dev \
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/* \
    #
    # Install docker binary
    && curl -L https://download.docker.com/linux/static/stable/x86_64/docker-19.03.9.tgz | tar xvz docker/docker \
    && cp docker/docker /usr/local/bin \
    && rm -R docker \
    # 
    # Install Spark
    && pip install --disable-pip-version-check --no-cache-dir pyspark==2.4.6 \
    #
    # Install Pin
    && sh -c 'curl https://software.intel.com/sites/landingpage/pintool/downloads/pin-${PIN_VERSION}-gcc-linux.tar.gz | tar -xvz -C /opt' \
    && ln -s $PIN_HOME/pin /usr/local/bin \
    #
    # Install PARSEC
    && sh -c 'curl -L http://parsec.cs.princeton.edu/download/${PARSEC_VERSION}/parsec-${PARSEC_VERSION}-core.tar.gz | tar -xvz -C /opt' \
    #
    # Create a non-root user to use if preferred
    && groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Install pip deps
RUN pip install --disable-pip-version-check --no-cache-dir \
    pandas==1.0.4 \
    pyarrow==0.17.1 \
    joblib==0.15.1 \
    pylint==2.5.3 \
    yapf==0.30.0 \
    scikit-learn==0.23.1 \
    jupyter==1.0.0 \
    matplotlib==3.2.2 \
    click==7.1.2 \
    debugpy==1.0.0b12

# Copy code in the container
COPY ./ /home/$USERNAME/dl-prefect/
RUN chown -R $USERNAME:$USERNAME /home/$USERNAME/dl-prefect/

# Set working directory
WORKDIR /home/$USERNAME/dl-prefect/

# Compile and install roitrace
RUN make PIN_ROOT=${PIN_HOME} -C src && ln -s ${PWD}/src/obj-intel64/roitrace.so /usr/local/bin

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog
