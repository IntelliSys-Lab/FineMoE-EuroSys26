#! /bin/bash

set -ex

export HUGGINGFACE_TOKEN=YOUR_HUGGINGFACE_TOKEN

apt-get update 
apt-get -y install software-properties-common ca-certificates python3 python3-pip python-is-python3 wget git vim net-tools jq zip tmux pciutils

pip install -e .

FLASH_ATTENTION_FORCE_BUILD=TRUE pip install flash-attn

huggingface-cli login --token $HUGGINGFACE_TOKEN --add-to-git-credential
