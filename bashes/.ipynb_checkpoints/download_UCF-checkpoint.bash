#!/bin/bash

wget https://crcv.ucf.edu/data/UCF101/UCF101.rar --no-check-certificate
unrar x UCF101.rar
rm UCF101.rar

mv UCF-101/* .
rm -r UCF-101
