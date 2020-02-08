#!/bin/bash

hostname=54.208.142.120
username=ec2-user
key_loc=~/.ssh/LightsailDefaultKey-us-east-1.pem

ssh -i $key_loc $username@$hostname
