#!/usr/bin/env bash

export LAMBDA_IP="<IP>"
scp -i ~/.ssh/lambda-labs-ssh-key.pem ~/.ssh/id_lambda_github* ubuntu@${LAMBDA_IP}:.ssh/

# Which region FS are we using?
export LAMBDA_REGION_FS="default-us-south-2"

# Anaconda Python 3.12 in a conda environment
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/$LAMBDA_REGION_FS/miniconda3

export PATH="$HOME/$LAMBDA_REGION_FS/miniconda3/bin:$PATH"
conda init bash
source ~/.bashrc

conda create -n eridu python=3.12 -y
conda activate eridu

pipx install poetry
poetry config virtualenvs.create false

# SSH Authetication
cat >> ~/.ssh/config <<'EOF'
Host github.com
  AddKeysToAgent yes
  IdentityFile ~/.ssh/id_lambda_github
EOF
eval "$(ssh-agent -p)"

# Install Java for PySpark ETL
sudo apt install openjdk-11-jre-headless -y

# Project specific stuff
cd $HOME/$LAMBDA_REGION_FS
git clone git@github.com:Graphlet-AI/eridu.git
cd eridu
poetry install


