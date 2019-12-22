#!/bin/bash

cd ../../../../
#docker run  -it -p 8080:8080  --mount type=bind,source="$(pwd)",target=/app mfeurer/auto-sklearn /bin/bash -c "cd app/NepalEarthquakes && pip install -e . && cd src/models && pwd && ls -a && python3 autosk_script.py" 
docker run  -it -p 8080:8080  --mount type=bind,source="$(pwd)",target=/app mfeurer/auto-sklearn /bin/bash -c "cd app/NepalEarthquakes && pip install -e . && cd src/models/autosk_scripts && pwd && ls -a && python3 autosk_script.py" 
