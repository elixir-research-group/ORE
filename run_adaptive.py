import pyterrier as pt
import pyterrier_alpha as pta
from ir_measures import nDCG, R
from baselines.gar import GAR
from baselines.quam import QUAM
from ore_adaptive import OREAdaptive

from pyterrier_dr import FlexIndex, TasB, TctColBert, NumpyIndex

from pyterrier_t5 import MonoT5ReRanker


import torch

import argparse

import pandas as pd
import random
random.seed(42)


parser = argparse.ArgumentParser()
parser.add_argument("--lk", type=int, default=16, help="the value of k for selecting k neighbourhood graph")
parser.add_argument("--dl", type=int, default=19, help="dl 19 or 20")
parser.add_argument("--budget", type=int, default=100, help="budget c")
parser.add_argument("--ce", type=int, default=7, help="number of cross encoder calls")
parser.add_argument("--s", type=int, default=30, help="top s docs (S) to calculate the set affinity.")
parser.add_argument("--verbose", action="store_true", help="if show progress bar.")

parser.add_argument("--s1", type=int, default=25, help="top s1 docs")
parser.add_argument("--s2", type=int, default=15, help="top s2 docs")

args = parser.parse_args()

dataset = pt.get_dataset('irds:msmarco-passage')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"device: {device}")

model = TasB.dot(batch_size=1, device=device)
idx = FlexIndex('<path to tasb flex index>')  # default path "indices/msmarco-passage.tasb.flex"

# indexref  =  "<path to bm25 index>"
# existing_index = pt.IndexFactory.of(indexref)

# retriever = pt.terrier.Retriever(existing_index, wmodel="BM25")

retriever = pt.terrier.Retriever.from_dataset('msmarco_passage', 'terrier_stemmed', wmodel='BM25', num_results=100)

tct_retriever = ( TctColBert('castorini/tct_colbert-v2-hnp-msmarco') >>
                                            NumpyIndex('<path to tct index>', verbose=False, cuda=False))  # default path "indices/castorini_tct_colbert-v2-hnp-msmarco"


bm25_cerberus = pt.terrier.Retriever.from_dataset('msmarco_passage', 'terrier_stemmed', wmodel='BM25', num_results=100)



"""
We re-use the existing corpus graph (introduced in GAR paper) and laff graph (introduced in Quam paper).
"""

graph = pta.Artifact.from_hf('macavaney/msmarco-passage.corpusgraph.bm25.16')    
laff_graph = pta.Artifact.from_hf('macavaney/msmarco-passage.corpusgraph.bm25.128.laff').to_limit_k(args.lk)

scorer = pt.text.get_text(dataset, 'text') >> MonoT5ReRanker(verbose=False, batch_size=args.batch)


cerberus = OREAdaptive(model, scorer,idx, graph, laff_graph, budget=args.budget, verbose=True)

dataset = pt.get_dataset(f'irds:msmarco-passage/trec-dl-20{args.dl}/judged')

pd.set_option('display.max_columns', None) 
pd.set_option('display.width', None)


save_dir=f"runs/dl{args.dl}/{args.graph_name}/"

print(f"number of cross encoder calls: {args.ce}")


tct_res = tct_retriever(dataset.get_topics())


result = pt.Experiment(
        [   
            retriever % args.budget >> scorer,

            retriever  >> GAR(scorer, graph, num_results=args.budget),


            retriever >>  QUAM(scorer=scorer,corpus_graph = laff_graph, num_results=args.budget,
                              cross_enc_budget=args.ce, top_k_docs=args.s, batch_size=args.batch,
                                  verbose=args.verbose),

            bm25_cerberus >> OREAdaptive(model, scorer, idx, graph,laff_graph, budget=args.budget, 
                                      cross_enc_budget=args.ce, param_bounds = (0.25,0.95), num_bm25_calls=0,
                                      verbose=False, top_s=args.s1, top_s2=args.s2)

            ],
        dataset.get_topics(),
        dataset.get_qrels(),
        [nDCG@10, nDCG@args.budget, R(rel=2)@args.budget],
        names=[
            f"{args.retriever}_monot5.c{args.budget}",
            f"GAR.c{args.budget}",
            f"QuAM.c{args.budget}",
            f"ore.c{args.budget}"
            ],
            save_dir=save_dir,
            save_mode='reuse'
    )

print(result.T)


