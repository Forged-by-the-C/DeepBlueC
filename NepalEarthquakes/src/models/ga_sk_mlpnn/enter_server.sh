#!/bin/bash

hostname=ec2-54-84-204-128.compute-1.amazonaws.com
username=ec2-user
key_loc=~/.ssh/nva.pem

ssh -i $key_loc $username@$hostname
