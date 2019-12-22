#!/bin/bash

hostname=ec2-54-235-58-4.compute-1.amazonaws.com
username=ec2-user
key_loc=~/.ssh/nva.pem

ssh -i $key_loc $username@$hostname
