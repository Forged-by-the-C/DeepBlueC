#!/bin/bash

hostname=54.208.142.120
username=ec2-user
key_loc=~/.ssh/LightsailDefaultKey-us-east-1.pem
proj_dir=NepalEarthquakes
file_dir=../../../$proj_dir
#subdirs=(src data requirements.txt setup.py)
subdirs=(src)

for i in "${subdirs[@]}"
do
    scp -i $key_loc -r $file_dir/$i $username@$hostname:~/$proj_dir
done

