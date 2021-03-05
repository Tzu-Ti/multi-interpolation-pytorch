#!/bin/bash

pip install -r requirements/requirements.txt
python requirements/imageio_download_ffmpeg.py
pip install imageio-ffmpeg