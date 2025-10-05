#!/bin/bash
conda remove -n bioinfo --all -y
conda create -n bioinfo python=3.10 -y

conda activate bioinfo

pip install torch==2.0.0

pip install numpy==1.24.2 scipy==1.10.0 pandas==1.5.3
pip install -e .
