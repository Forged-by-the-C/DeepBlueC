#!/bin/bash

sudo yum update -y
sudo yum install -y docker
sudo service docker start
sudo docker pull mfeurer/auto-sklearn
sudo docker ps -a
