  
# Start from ubuntu container
FROM tensorflow/tensorflow:2.2.0-jupyter

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
ENV PARSECPLAT=${OSTYPE}-${HOSTTYPE}
ENV LD_LIBRARY_PATH=${PARSECDIR}/pkgs/libs/hooks/inst/${PARSECPLAT}/lib

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
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/* \
    #
    # Install pip deps
    && pip --disable-pip-version-check --no-cache-dir install tensorflow-probability \
    #
    # Download pin and compile tools
    && sh -c 'curl https://software.intel.com/sites/landingpage/pintool/downloads/pin-${PIN_VERSION}-gcc-linux.tar.gz | tar -xvz -C /opt' \
    && make -C ${PIN_HOME}/source/tools/MemTrace \
    && make -C ${PIN_HOME}/source/tools/ManualExamples \
    && ln -s $PIN_HOME/pin /usr/local/bin/ \
    && ln -s $PIN_HOME/source/tools/MemTrace/obj-intel64/*.so /usr/local/bin/ \
    && ln -s $PIN_HOME/source/tools/ManualExamples/obj-intel64/*.so /usr/local/bin/ \
    #
    # Install PARSEC
    && sh -c 'curl -L http://parsec.cs.princeton.edu/download/${PARSEC_VERSION}/parsec-${PARSEC_VERSION}-core.tar.gz | tar -xvz -C /opt' \
    #
    # Create a non-root user to use if preferred
    && groupadd --gid $USER_GID $USERNAME \
    && useradd -s /bin/bash --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME\
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Copy code in the container
COPY ./ /home/$USERNAME/dl-prefect/
RUN chown -R $USERNAME:$USERNAME /home/$USERNAME/dl-prefect/

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog

# Set working directory
WORKDIR /home/$USERNAME/dl-prefect/