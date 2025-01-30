import pyterrier as pt
import pyterrier_alpha as pta
from ir_measures import nDCG, R
from baselines.hybrid_cc import HybridCC
from baselines.hybrid_rrf import HybridRRF
from ore_hybrid import OREHybrid

from pyterrier_dr import FlexIndex, TasB, TctColBert, NumpyIndex

from pyterrier_t5 import MonoT5ReRanker

import torch
import argparse

import pandas as pd
import random


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
idx = FlexIndex.from_hf('<path to tasb flex index>')


indexref  =  "<path to bm25 index>"
existing_index = pt.IndexFactory.of(indexref)

retriever = pt.terrier.Retriever(existing_index, wmodel="BM25")
terrier_ = pt.terrier.Retriever(existing_index, wmodel="BM25", num_results=256)


tct_retriever = ( TctColBert('castorini/tct_colbert-v2-hnp-msmarco') >>
                                            NumpyIndex('<path to tct index>', verbose=False, cuda=False))



bm25_cerberus = pt.terrier.Retriever(existing_index, wmodel="BM25", num_results=args.budget) 



"""
We re-use the existing bm25 and tct-based corpus graphs (introduced in GAR paper) and laff graph (introduced in Quam paper).
"""

graph = pta.Artifact.from_hf('macavaney/msmarco-passage.corpusgraph.bm25.16')  
tct_graph = pta.Artifact.load('macavaney/msmarco-passage.corpusgraph.tcthnp.16')

laff_graph = pta.Artifact.from_hf('macavaney/msmarco-passage.corpusgraph.bm25.128.laff').to_limit_k(args.lk)



scorer = pt.text.get_text(dataset, 'text') >> MonoT5ReRanker(verbose=False, batch_size=args.batch)


dataset = pt.get_dataset(f'irds:msmarco-passage/trec-dl-20{args.dl}/judged')

pd.set_option('display.max_columns', None) 
pd.set_option('display.width', None)


save_dir=f"runs/hybrid/dl{args.dl}/{args.graph_name}/"

print(f"number of cross encoder calls: {args.ce}")


tct_res = tct_retriever(dataset.get_topics())



result = pt.Experiment(
    [   
        retriever >> HybridRRF(scorer, num_results=args.budget, df2 = tct_res,verbose=args.verbose),

        retriever >> HybridCC(scorer, num_results=args.budget, df2 = tct_res,verbose=args.verbose),

        bm25_cerberus >> OREHybrid(scorer, num_results=args.budget, laff_graph = tct_graph, dense_retriever = tct_retriever,
                                                terrier_index = existing_index, bm25_retriever = terrier_ ,cross_enc_budget=args.ce ,verbose=args.verbose)

        ],
    dataset.get_topics(),
    dataset.qrels,
    [nDCG@10, nDCG@args.budget, R(rel=2)@args.budget],
    names=[
        f"rrf_bm25_tct.c{args.budget}",
        f"cc_bm25_tct.c{args.budget}",
        f"Hybridore.c{args.budget}"
        ],
        save_dir=save_dir,
        save_mode='reuse'


)

print(result.T)

