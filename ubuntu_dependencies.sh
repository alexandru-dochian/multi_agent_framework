#!/bin/bash

# Install dependencies
poetry env use 3.11

poetry shell

poetry install

cd crazyflie-lib-python && pip install -e . && cf ..

python3 -m pip install pyqt5

python3 -m pip install cfclient
