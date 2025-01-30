from collections import Counter
from typing import List
from statistics import mean
import torch
import numpy as np
from itertools import chain
from scipy.special import softmax
from joblib import Parallel
import time
import heapq
import scipy
import random
from collections import defaultdict

random.seed(42)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta
from pyterrier_adaptive import CorpusGraph
import ir_datasets
import torch





dataset_store = ir_datasets.load('msmarco-passage')
docstore = dataset_store.docs_store()

indexref  =  "<path to bm25 index>"
existing_index = pt.IndexFactory.of(indexref)
ret_scorer = pt.text.scorer(takes="docs", body_attr="text", wmodel="BM25",background_index= existing_index, controls={"termpipelines": "Stopwords,PorterStemmer"})


class OREAdaptive(pt.Transformer):
    def __init__(self, dual_encoder, scorer,corpus_index, graph: CorpusGraph, 
                 laff_graph: CorpusGraph, *, budget: int = 100, verbose: bool = False, 
                 batch_size : int = 16, num_bm25_calls: int=10, top_s: int = 25, top_s2: int = 15, cross_enc_budget: int = 2, 
                 param_bounds: tuple = (0.25,0.9)):
        self.scorer = scorer
        self.graph = graph
        self.laff_graph = laff_graph
        self.budget = budget
        self.verbose = verbose
        self.corpus_index = corpus_index
        self.batch_size = batch_size
        self.num_bm25_calls = num_bm25_calls
        self.cross_enc_budget = cross_enc_budget
        self.top_s = top_s
        self.top_s2 = top_s2
        self.param_bounds = param_bounds
        self.dual_encoder = dual_encoder

    def estimate_bm25_score_batch(self, qids,queries,docids, initial_results):
            batch = []
            for qid,query,docid in zip(qids,queries,docids):
                batch.append([qid,query,docid,docstore.get(docid).text])

            df = pd.DataFrame(batch,columns=["qid", "query", "docno", "text"])
            result_df = ret_scorer(df)

            return list(result_df["docno"].values), list(result_df["score"].values)
    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        result_builder = pta.DataFrameBuilder(['qid', 'query', 'docno', 'score', 'rank'])
        groups = list(inp.groupby('query'))
        lambda_param=0.65
        lambda_param_1 = 0.45
        lambda_param_2 = 0.65

        
        for i, (query, initial_results) in enumerate(groups):


            qid = initial_results['qid'].iloc[0]

            initial_results = initial_results.sort_values('score', ascending=False)
            
            arms = [Arm(docid, name='initial_results_'+docid) for docid in initial_results['docno'].tolist()[:self.budget]]


            results = {}
            bm25_scores = dict(zip(initial_results["docno"].values,initial_results["score"].values))
            
            count = 0
            while len(arms) > 0 and len(results) < self.budget:
                if count ==0:
                    arm = sorted(arms, key = lambda x: x.estimate_utility(), reverse=True)[:self.batch_size]
                else:
                    cluster_heads = [doc for doc, _ in Counter(results).most_common(self.top_s)]

                    # Step 2: Collect neighbors and scores efficiently
                    cluster_neigh_lookup = defaultdict(list)

                    for cluster_head in cluster_heads:
                        # Fetch neighbors and weights once per cluster_head
                        neighbors, scores = self.laff_graph.neighbours(cluster_head, weights=True)
                        
                        # Use defaultdict to accumulate scores
                        for neighbor, score in zip(neighbors, scores):
                            cluster_neigh_lookup[neighbor].append(score)
                        

                    # Step 3: Compute mean scores
                    cluster_neigh_lookup = {key: mean(scores) for key, scores in cluster_neigh_lookup.items()}
                    filtered_arms = [arm for arm in arms if arm.docnos[-1] not in results]

                    if len(bm25_scores)<(len(initial_results)+self.num_bm25_calls):
                        donos_missing = [x.docnos[-1] for x in filtered_arms if x.docnos[-1] not in bm25_scores ]
                        if len(donos_missing)>0:
                                qids = len(donos_missing)* [qid]
                                queries = len(donos_missing)* [query]
                                bm25_time = time.time()
                                docnos,scores = self.estimate_bm25_score_batch(qids,queries,donos_missing,initial_results)
                                bm25_scores = dict(zip(docnos,scores))

                    if prev_heads == cluster_heads:
                        start =time.time()
                        neighbor_criteria_arms = [arm for arm in filtered_arms if arm.docnos[-1] in cluster_neigh_lookup]

                        # Pre-compute scores instead of recalculating them multiple times
                        criteria_scores = [(arm, (cluster_neigh_lookup.get(arm.docnos[-1], 0))) for arm in neighbor_criteria_arms]

                        # Use heapq.nlargest to get top 15 arms by laff scores (avoids full sorting)
                        new_arms = [arm for arm, _ in heapq.nlargest(35, criteria_scores, key=lambda x: x[1])]

                        # Filter remaining arms in a single pass
                        remaining_arms = [(arm, bm25_scores.get(arm.docnos[-1], 0)) for arm in neighbor_criteria_arms if  arm.docnos[-1] in bm25_scores and arm.docnos[-1] not in new_arms ]

                        # Use heapq.nlargest to get top 10 BM25 arms
                        bm25_arms = [arm for arm, _ in heapq.nlargest(25, remaining_arms, key=lambda x: x[1])]

                        # Combine new_arms and bm25_arms
                        new_arms.extend(bm25_arms)

                        if len(new_arms) ==0:
                            new_arms = filtered_arms

                        arm = sorted(new_arms, key=lambda x: x.cer_scores[x.docnos[-1]] if x.docnos[-1] in list(x.cer_scores.keys()) else x.estimate_cer_score(qid,query,x.docnos,results,self.graph, self.laff_graph, initial_results,bm25_scores,cluster_heads,lambda_param,lambda_param_1,lambda_param_2,cluster_neigh_lookup),reverse=True)[:self.batch_size]
                        

                    else:
                        neighbor_criteria_arms = [arm for arm in filtered_arms if arm.docnos[-1] in cluster_neigh_lookup ]

                        # Pre-compute scores instead of recalculating them multiple times
                        criteria_scores = [(arm, cluster_neigh_lookup.get(arm.docnos[-1], 0)) for arm in neighbor_criteria_arms]

                        # Use heapq.nlargest to get top 15 arms by laff scores (avoids full sorting)
                        new_arms = [arm for arm, _ in heapq.nlargest(35, criteria_scores, key=lambda x: x[1])]

                        # Filter remaining arms in a single pass
                        remaining_arms = [(arm, bm25_scores.get(arm.docnos[-1], 0)) for arm in neighbor_criteria_arms if arm.docnos[-1] in bm25_scores and arm.docnos[-1] not in new_arms]

                        # Use heapq.nlargest to get top 10 BM25 arms
                        bm25_arms = [arm for arm, _ in heapq.nlargest(25, remaining_arms, key=lambda x: x[1])]

                        # Combine new_arms and bm25_arms
                        new_arms.extend(bm25_arms)
                        new_arms = set(new_arms)
                        if len(new_arms) ==0:
                            new_arms = filtered_arms


                        cer_scores = [x.estimate_cer_score(qid,query,x.docnos,results,self.graph, self.laff_graph,  initial_results,bm25_scores,cluster_heads, lambda_param,lambda_param_1,lambda_param_2,cluster_neigh_lookup) for x in new_arms]
                        
                        arm = sorted(zip(new_arms,cer_scores), key= lambda x: x[1], reverse=True)[:self.batch_size]
                        arm = [ x for x,_ in arm]

                docnos_final = [x.docnos[-1] for x in arm]
                all_docnos = [x.docnos[-1] for x in arms]


                

                if count>0:             

                   estimated_rank_scores = [x.cer_scores[x.docnos[-1]]   for x in arm if x.docnos[-1] in x.cer_scores]
                if len(results)<(min(self.batch_size*self.cross_enc_budget, self.budget)):
                    with torch.no_grad():
                        query_vecs = self.dual_encoder.encode_queries([query])[0].reshape(1,-1)
                    doc_object = [{"docno":docno} for docno in docnos_final]
                    doc_vecs = np.concatenate([ doc_vector.reshape(1,-1) for doc_vector in self.corpus_index.vec_loader()(pd.DataFrame(doc_object))["doc_vec"].values])
                    
                    dual_score = (query_vecs.dot(doc_vecs.T))[0]
                    batch = pd.DataFrame(docnos_final, columns=['docno'])
                    batch['qid'] = qid
                    batch['query'] = query
                    reranked_scores = list(self.scorer(batch)["score"].values)


                    ranked_set_scores = [x+score for x,score in zip(reranked_scores,dual_score)]#ranked_set_scores
                    if count >0:

                        bm25_features = np.array([x.bm25_scores[x.docnos[-1]]  if x.docnos[-1] in x.bm25_scores else 0 for x in arm ]).reshape(-1,1)
                        affinity_features = np.array([x.estimates[x.docnos[-1]]  for x in arm]).reshape(-1,1)


                        neighbor_score_features = np.array([x.cross_enc_avg[x.docnos[-1]] for x in arm]).reshape(-1,1)

                        features = np.concatenate((bm25_features,affinity_features,neighbor_score_features),axis=1)

                        params = scipy.optimize.lsq_linear(features, ranked_set_scores,lsq_solver="exact", bounds=(self.param_bounds))


                        lambda_param = params["x"][0]
                        lambda_param_1 = params["x"][1]
                        lambda_param_2 = params["x"][2]
                else:
                    ranked_set_scores = estimated_rank_scores



                for x,docno,score_value in zip(arm,docnos_final,ranked_set_scores):
                        results[docno] = score_value
                        x.push(score_value)

                if len(results)<self.budget:
                    S_2 = Counter(results).most_common(self.top_s2)
                    S2 = [doc[0] for doc in S_2]
                    neighbor_lookup = set(S2).intersection(set(docnos_final))
                    
                    for docno in neighbor_lookup:
                            neighbors = self.laff_graph.neighbours(docno)
                            for neighbor in neighbors.tolist()[:16]:
                                if neighbor not in all_docnos: 
                                    neighbor_arm = Arm(neighbor, name=f'neighbors_{docno}')
                                    neighbor_arm.push(score_value)
                                    arms.append(neighbor_arm)
                                    all_docnos.append(neighbor)

                if count >0:
                    prev_heads = cluster_heads
                else:
                    prev_heads=[]
                count+=1

                arms = [a for a in arms if not a.is_exhausted()]
        

            for rank, (docno, final_score) in enumerate(Counter(results).most_common()):
                result_builder.extend({
                    'qid': qid,
                    'query': query,
                    'docno': docno,
                    'score': final_score,
                    'rank': rank,
                })

        return result_builder.to_df()


class Arm:
    def __init__(self, docnos: str, name: str = ''):
        self.docnos = [docnos]
        self.scores = []
        self.estimated_scores = []
        self.name = name
        self.cer_scores = {}
        self.bm25_scores = {}
        self.laff_scores = {}
        self.estimates = {}
        self.cross_enc_avg = {}

    def is_exhausted(self):
        return len(self.docnos) == 0

    def pull(self):
        assert not self.is_exhausted()
        next_docno, self.docnos = self.docnos[0], []
        return next_docno

    def push(self, score: float):
        self.scores.append(score)

    def estimate_utility(self):
        # for now: mean of scores
        # TODO: something fancy
        if len(self.scores) == 0:
            return float('-inf')
        return sum(self.scores) / len(self.scores)

    def estimate_cer_score(self, qid,query, docnos, results,neigh_graph, graph,initial_results,bm25_score_dict,cluster_heads,lambda_param,lambda_1,lambda_2,cluster_neigh_lookup):
        doc= docnos[-1]
        if doc in bm25_score_dict:
            self.bm25_scores[doc] = bm25_score_dict[doc]
            bm25_score = bm25_score_dict[doc]
        else:
            bm25_score = 0


        laff_score_dict = dict(zip(*graph.neighbours(doc,weights=True)))
        crss_enc_scores = []
        
        valid_cluster_heads = [res for res in cluster_heads if res in laff_score_dict]
        crss_enc_scores.extend(results[res] for res in valid_cluster_heads)

        
        self.estimated_scores = lambda_param * bm25_score  + lambda_1 * (cluster_neigh_lookup.get(doc,0))

        if len(crss_enc_scores)>0:
            score_utility = sum(crss_enc_scores)/len(crss_enc_scores)
        else:
            score_utility = self.estimate_utility()
            if score_utility==float("-inf"):
                score_utility=0
        self.cross_enc_avg[docnos[-1]]=score_utility
        self.estimates[docnos[-1]]= self.estimated_scores
        self.cer_scores[docnos[-1]] = (lambda_1 *  self.estimated_scores )+ (lambda_2) * (score_utility)
        return  (lambda_1 *  self.estimated_scores )+ (lambda_2 * score_utility) 




    def estimate_bm25_score(self, query_id,query, doc_id):
        df = pd.DataFrame([[query_id,query,doc_id,docstore.get(doc_id).text]],columns=["qid", "query", "docno", "text"])
        return ret_scorer(df)["score"].values