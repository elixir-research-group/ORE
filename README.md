# Breaking the Lens of the Telescope: Online Relevance Estimation over Large Retrieval Sets


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


