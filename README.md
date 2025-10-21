# KV Cache Eviction
This repository contains the code and data artifacts required to reproduce the results reported in our paper submission.  
The implementation is based on the open-source repository [KVCache-Factory](https://github.com/Zefan-Cai/KVCache-Factory), with modifications and extensions to support the proposed adaptive block size KV cache compression method.


## Requirements
- Dependencies are listed in `requirements.txt`
To install all dependencies:
```bash
pip install -r requirements.txt
```

## Reproducing Results
The main scripts for running experiments are located in the scripts/ folder.
For example, to run LongBench:
```bash
bash scripts/scripts_longBench/eval.sh
```
To run NIAH:
```bash
bash scripts/scripts_needle/eval.sh
```
