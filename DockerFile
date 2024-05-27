FROM gocd/gocd-agent-ubuntu-22.04:v24.1.0

USER root

ENV DEBIAN_FRONTEND=noninteractive

RUN echo "***** install python *****" && \
    apt-get update -y && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get install -y python3.8 python3-pip && \
    pip install kfp==2.7.0 boto3==1.24.28 && \
    python3 -V && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install kubectl
RUN echo "**** Install kubectl ****" && \
   curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl" && \
   install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl

USER go
