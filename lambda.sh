#!/usr/bin/env bash

#
# Configuration and Github SSH key setup
#

# What is the IP address of the Lambda Labs Ubuntu machine?
export LAMBDA_IP="<YOUR_LAMBDA_IP_ADDRESS>"

# Which region FS are we using?
export LAMBDA_REGION_FS="default-us-south-2"

# Lambda Labs SSH key
export LAMBDA_LABS_KEY="lambda-labs-ssh-key.pem"

# Copy over Github SSH keys
scp -i ~/.ssh/${LAMBDA_LABS_KEY} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ~/.ssh/id_lambda_github* ubuntu@${LAMBDA_IP}:.ssh/


#
# SSH over to the Lambda Labs machine. Run the rest of the code below locally there.
#
# TODO: Fix this so the script below runs from this SSH command.
#

ssh -i ~/.ssh/${LAMBDA_LABS_KEY} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ubuntu@${LAMBDA_IP}

# Do everything on the persistent filesystem
cd ${HOME}/${LAMBDA_REGION_FS}

# Update apt
sudo apt update -y

# Install Java for PySpark ETL
sudo apt install openjdk-11-jre-headless -y


#
# Anaconda Python 3.12 in a conda environment
#

curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
export MINICONDA_HOME="${HOME}/${LAMBDA_REGION_FS}/miniconda3"
./Miniconda3-latest-Linux-x86_64.sh -b -p "${MINICONDA_HOME}"

export PATH="${MINICONDA_HOME}/bin:${PATH}"
conda init bash
source ~/.bashrc

# Accept Anaconda TOS
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create the project's _environment_
conda create -n eridu python=3.12 -y
conda activate eridu

# Install poetry and disable virtualenvs - we have conda
sudo apt install pipx -y
pipx install --force poetry
poetry config virtualenvs.create false

#
# Github SSH authentication and code checkout
#
cat >> ~/.ssh/config <<'EOF'
Host github.com
  AddKeysToAgent yes
  StrictHostKeyChecking no
  IdentityFile ~/.ssh/id_lambda_github
EOF
eval "$(ssh-agent -s)"

# Clone the project repository and install its dependencies
cd ${HOME}/${LAMBDA_REGION_FS}
git clone git@github.com:Graphlet-AI/eridu.git
cd eridu
poetry install

echo 'export PATH=/home/ubuntu/.local/bin:$PATH' >> ~/.bashrc
source ~/.bashrc
