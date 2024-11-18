import pyterrier as pt
import pyterrier_alpha as pta
from ir_measures import nDCG, R
from pyterrier_adaptive import GAR

from cerberus import Cerberus


bm25 = pta.Artifact.from_hf('macavaney/msmarco-passage.pisa').bm25(num_results=100)
scorer = pta.Artifact.from_hf('macavaney/msmarco-passage.monot5-base.cache')
graph = pta.Artifact.from_hf('macavaney/msmarco-passage.corpusgraph.bm25.16')
gar = GAR(scorer, graph, num_results=100)
cerberus = Cerberus(scorer, graph, budget=100, verbose=True)
dataset = pt.get_dataset('irds:msmarco-passage/trec-dl-2019/judged')


print(pt.Experiment(
    [bm25, bm25 >> scorer, bm25 >> gar, bm25 >> cerberus],
    dataset.get_topics(),
    dataset.get_qrels(),
    [nDCG@10, nDCG@100, R@100],
    names=['bm25', 'bm25 >> monot5', 'bm5 >> GAR(monot5)', 'bm25 >> Cerberus(monot5)']
))
