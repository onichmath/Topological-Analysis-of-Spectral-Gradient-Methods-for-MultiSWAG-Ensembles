#!/bin/bash
# python main.py --optimizer adam train
# python main.py --optimizer adamw train
# python main.py --optimizer muon train

python main.py --optimizer 10p train
python main.py --optimizer muon10p train
python main.py --optimizer muonspectralnorm train
python main.py --optimizer spectralnorm train
