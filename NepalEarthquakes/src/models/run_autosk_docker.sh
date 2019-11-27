#!/bin/bash

cd ../../../
#docker run  -it -p 8080:8080  --mount type=bind,source="$(pwd)",target=/app mfeurer/auto-sklearn /bin/bash -c "cd app/NepalEarthquakes && pip install -e . && jupyter notebook --ip=0.0.0.0 --port=8080 --allow-root" 
docker run  -it -p 8080:8080  --mount type=bind,source="$(pwd)",target=/app mfeurer/auto-sklearn /bin/bash -c "cd app/NepalEarthquakes && pip install -e . && cd src/models && pwd && ls -a && python3 autosk_script.py" 
