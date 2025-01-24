import pyterrier as pt
import pyterrier_alpha as pta
from ir_measures import nDCG, R
#from pyterrier_adaptive import GAR
from gar_aff import GAR
from hybrid_gar import HybridGAR
from pyterrier_quam import QUAM
from hybrid_quam import HybridQUAM
from pyterrier_dr import FlexIndex, TasB, TctColBert, NumpyIndex

from pyterrier_t5 import MonoT5ReRanker
from ranklamma_pyterrier import RankLammaReRanker


import torch

import argparse

import pandas as pd
import joblib
from cerberus import Cerberus
import time
import random
random.seed(42)


parser = argparse.ArgumentParser()
parser.add_argument("--lk", type=int, default=16, help="the value of k for selecting k neighbourhood graph")
parser.add_argument("--graph_name", type=str, default="gbm25", help="name of the graph")
parser.add_argument("--dl_type", type=int, default=19, help="dl 19 or 20")
parser.add_argument("--seed", type=int, help="seed",default=1234)
parser.add_argument("--batch", type=int, default=16, help="batch size")
parser.add_argument("--budget", type=int, default=100, help="budget c")
parser.add_argument("--ce", type=int, default=7, help="number of cross encoder calls")
parser.add_argument("--s", type=int, default=30, help="top s docs (S) to calculate the set affinity.")
parser.add_argument("--verbose", action="store_true", help="if show progress bar.")
parser.add_argument("--retriever", type=str, default="bm25", help="name of the retriever")

parser.add_argument("--s1", type=int, default=25, help="top s1 docs")
parser.add_argument("--s2", type=int, default=15, help="top s2 docs")

args = parser.parse_args()

dataset = pt.get_dataset('irds:msmarco-passage')


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"device: {device}")

model = TasB.dot(batch_size=1, device=device)
idx = FlexIndex.from_hf('macavaney/msmarco-passage.tasb.flex')

indexref  =  "/home/rathee/mabgar/indices/bm25_msmarco_passage"
existing_index = pt.IndexFactory.of(indexref)

if args.retriever == "bm25":
    retriever = pt.terrier.Retriever(existing_index, wmodel="BM25")

    tct_retriever = ( TctColBert('castorini/tct_colbert-v2-hnp-msmarco') >>
                                            NumpyIndex('/home/rathee/llmgar/indices/castorini__tct_colbert-v2-hnp-msmarco.np', verbose=False, cuda=False))

else:
    retriever = ( TctColBert('castorini/tct_colbert-v2-hnp-msmarco') >>
                                            NumpyIndex('/home/rathee/llmgar/indices/castorini__tct_colbert-v2-hnp-msmarco.np', verbose=False, cuda=False))

bm25_cerberus = pt.terrier.Retriever(existing_index, wmodel="BM25", num_results=args.budget) 



if args.graph_name == "gbm25":
    graph = pta.Artifact.from_hf('macavaney/msmarco-passage.corpusgraph.bm25.16')
else:
    graph = pta.Artifact.load('/home/rathee/quam/corpusgraph_k128').to_limit_k(args.lk)
    graph_128 = pta.Artifact.load('/home/rathee/quam/corpusgraph_k128')



## start with a corpus graph >> generate a laff graph (128 are saved) 


laff_graph = pta.Artifact.from_hf('macavaney/msmarco-passage.corpusgraph.bm25.128.laff').to_limit_k(args.lk)

scorer = pt.text.get_text(dataset, 'text') >> MonoT5ReRanker(verbose=False, batch_size=args.batch)

#scorer = pt.text.get_text(dataset, 'text') >> RankLammaReRanker(verbose=False, batch_size=16)

cerberus = Cerberus(model, scorer,idx, graph,laff_graph, budget=args.budget, verbose=True)

dataset = pt.get_dataset(f'irds:msmarco-passage/trec-dl-20{args.dl_type}/judged')

pd.set_option('display.max_columns', None) 
pd.set_option('display.width', None)


# save_dir=f"saved_pyterrier_runs/{args.graph_name}/dl{args.dl_type}/{args.retriever}/"

# if not os.path.exists(save_dir):
#     print("not found, creating folder.")
#     print(save_dir)
#     os.makedirs(save_dir)
#     exit()


# linear = 0.1 * retriever + 0.9 * tct_retriever

# query = "clustering hyptothesis"

# res = linear.search(query)

# print(res)

# res = tct_retriever.transform(pd.DataFrame({"qid": [1], "query": [query]}))
# print(res)


print(f"number of cross encoder calls: {args.ce}")



tct_res = tct_retriever(dataset.get_topics())


for i in range(6):
    result = pt.Experiment(
        [   
            #retriever % args.budget >> scorer,



            # retriever  >> GAR(scorer, laff_graph, num_results=args.budget),


            # retriever >>  QUAM(scorer=scorer,corpus_graph = laff_graph, num_results=args.budget,
            #                  cross_enc_budget=args.ce, top_k_docs=args.s, batch_size=args.batch,
            #                      verbose=args.verbose),

            bm25_cerberus >> Cerberus(model, scorer, idx, graph,laff_graph, budget=args.budget, 
                                      cross_enc_budget=args.ce, param_bounds = (0.25,0.95), num_bm25_calls=0,
                                      verbose=False, top_s=args.s1, top_s2=args.s2)


            # bm25_cerberus >> HybridAdaptiveCerberus(model, scorer, idx, graph,laff_graph, budget=args.budget, 
            #                           cross_enc_budget=args.ce, df2=tct_res, param_bounds = (0.25,0.95), num_bm25_calls=0,
            #                           verbose=False, top_s=args.s1, top_s2=args.s2)

            ],
        dataset.get_topics(),
        dataset.get_qrels(),
        [nDCG@10, nDCG@args.budget, R(rel=2)@args.budget],
        names=[
            #f"{args.retriever}_monot5.c{args.budget}",
            #f"GAR.c{args.budget}",
            #f"GAR_Laff.c{args.budget}",
            #f"QuAM.c{args.budget}",
            f"Cerberus.c{args.budget}"
            ]
    )

    print(result.T)

exit()


cerberus = Cerberus(model, scorer,idx, graph,laff_graph, budget=100, cross_enc_budget=1,param_bounds = (0.25,0.9),num_bm25_calls=100, verbose=True)
dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2020/judged')

pd.set_option('display.max_columns', None) 
pd.set_option('display.width', None)
print(pt.Experiment(
    [bm25_cerberus >> cerberus],
    dataset.get_topics(),
    dataset.get_qrels(),
    [nDCG@10, nDCG@100,nDCG@100, R@100, R(rel=2)@100],
    names=['bm25 >> Cerberus(monot5)']
))



cerberus = Cerberus(model, scorer,idx, graph,laff_graph, budget=100, num_bm25_calls=30,param_bounds=(0.3,0.85), verbose=True)
dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2019/judged')

pd.set_option('display.max_columns', None) 
pd.set_option('display.width', None)
print(pt.Experiment(
    [bm25_cerberus >> cerberus],
    dataset.get_topics(),
    dataset.get_qrels(),
    [nDCG@10, nDCG@100,nDCG@100, R@100, R(rel=2)@100],
    names=['bm25 >> Cerberus(monot5)']
))

cerberus = Cerberus(model, scorer,idx, graph,laff_graph, budget=100, num_bm25_calls=30,param_bounds=(0.3,0.85), cross_enc_budget=1, verbose=True)
dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2019/judged')

pd.set_option('display.max_columns', None) 
pd.set_option('display.width', None)
print(pt.Experiment(
    [bm25_cerberus >> cerberus],
    dataset.get_topics(),
    dataset.get_qrels(),
    [nDCG@10, nDCG@100,nDCG@100, R@100, R(rel=2)@100],
    names=['bm25 >> Cerberus(monot5)']
))