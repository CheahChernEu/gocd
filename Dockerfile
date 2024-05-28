FROM gocd/gocd-agent-ubuntu-22.04:v24.1.0

USER root

# Set non-interactive frontend to avoid prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.8 and pip
RUN echo "***** Install Python *****" && \
    apt-get update -y && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y python3.8 python3-pip && \
    # Clean up to reduce image size
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip install kfp==2.7.0 boto3==1.24.28

# Install kubectl
RUN echo "**** Install kubectl ****" && \
    curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && \
    install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

# Switch back to the go user
USER go
