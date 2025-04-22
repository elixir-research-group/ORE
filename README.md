# Breaking the Lens of the Telescope: Online Relevance Estimation over Large Retrieval Sets

This is the official Github repository for our paper [Breaking the Lens of the Telescope: Online Relevance Estimation over Large Retrieval Sets](https://arxiv.org/pdf/2504.09353)  accepted in 48th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR 2025), Padua, Italy, 13-17 July 2025,


## Requirements


```
pip install -r requirements.txt
pip install --upgrade git+https://github.com/terrierteam/pyterrier_dr.git
pip install --upgrade git+https://github.com/terrierteam/pyterrier_t5.git
```


## Reproducibility 

### Build index
To build the tasb and tct indexes, please run the following python file

```
python3 build_indexes.py
```

Please use the correct path of indexes in run_hybrid.py and run_adaptive.py

### run hybrid setting

After building the indexes, to reproduce the ORE hybrid, run the following command:

for budget 50

```
python3 run_hybrid.py --budget 50 --dl 19 --ce 4 
```

for budget 100

```
python3 run_hybrid.py --budget 100 --dl 19 --ce 7 
```

### run adaptive setting

for budget 50

```
python3 run_adaptive.py --budget 50 --s 10 --dl 19 --ce 4 --s1 10 --s2 15
```

for budget 100

```
python3 run_adaptive.py --budget 100 --s 30 --dl 19 --ce 7 --s1 25 --s2 15
```

## Citation
```
@article{rathee2025breaking,
  title={Breaking the Lens of the Telescope: Online Relevance Estimation over Large Retrieval Sets},
  author={Rathee, Mandeep and MacAvaney, Sean and Anand, Avishek and others},
  journal={arXiv preprint arXiv:2504.09353},
  year={2025}
}
```
