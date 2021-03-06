#!/usr/bin/env bash

# This script shows how to build the Docker image and push it to ECR to be ready for use
# by SageMaker.

# The argument to this script is the image name. This will be used as the image on the local
# machine and combined with the account and region to form the repository name for ECR.

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# set region
region=
if [ "$#" -eq 1 ]; then
    region=$1
else
    echo "usage: $0 <aws-region>"
    exit 1
fi
  


# Get the account number associated with the current IAM credentials
account=$(aws sts get-caller-identity --query Account --output text)

if [ $? -ne 0 ]
then
    exit 255
fi




# Build the docker image locally with the image name and then push it to ECR
# with the full name.

# Get the login command from ECR and execute it directly
$(aws ecr get-login --no-include-email --region us-west-2  --registry-ids 763104351884)

docker pull 763104351884.dkr.ecr.us-west-2.amazonaws.com/tensorflow-training:2.3.1-gpu-py37
