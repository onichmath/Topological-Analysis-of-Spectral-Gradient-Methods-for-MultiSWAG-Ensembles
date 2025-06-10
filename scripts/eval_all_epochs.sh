#!/bin/bash
python main.py --optimizer adam eval
python main.py --optimizer adamw eval
python main.py --optimizer muon eval
python main.py --optimizer 10p eval
python main.py --optimizer muon10p eval
python main.py --optimizer muonspectralnorm eval
python main.py --optimizer spectralnorm eval