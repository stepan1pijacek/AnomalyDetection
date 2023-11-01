#!/bin/sh
export FLASK_APP=./AnomalyDetectionService/index.py

flask --debug run -h 0.0.0.0 -p 6000