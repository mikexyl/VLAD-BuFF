FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04


# Install required tools (no Pixi yet)
RUN apt-get update && apt-get install -y curl unzip git

# Add build argument for username
ARG USERNAME

# Install sudo, create user, add to sudoers with no password
RUN apt-get update && apt-get install -y sudo \
    && useradd -m -s /bin/bash "$USERNAME" \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# Install Pixi and put it into /usr/local/bin
RUN curl -fsSL https://pixi.sh/install.sh | PIXI_HOME=/usr/local bash 

