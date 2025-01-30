from typing import Optional
import numpy as np
from collections import Counter, defaultdict
import pyterrier as pt
import pandas as pd
import ir_datasets
from sklearn.preprocessing import minmax_scale
logger = ir_datasets.log.easy()
import torch
from statistics import mean
import scipy



dataset = pt.get_dataset('irds:msmarco-passage-v2')
text_loader = pt.text.get_text(dataset, 'text')


class OREHybrid(pt.Transformer):

    def __init__(self,
        scorer: pt.Transformer,
        num_results: int = 100,
        laff_graph = None,
        cross_enc_budget: int = 7,
        top_s: int = 10,
        batch_size: Optional[int] = None,
        backfill: bool = True,
        verbose: bool = True,
        terrier_index = None,
        bm25_retriever = None,
        dense_retriever = None):

        self.scorer = scorer
        self.top_s = top_s
        self.laff_graph = laff_graph
        self.terrier_index = terrier_index
        self.bm25_retriever = bm25_retriever
        self.num_results = num_results
        self.cross_enc_budget = cross_enc_budget
        if batch_size is None:
            batch_size = scorer.batch_size if hasattr(scorer, 'batch_size') else 16
        self.batch_size = batch_size
        self.backfill = backfill
        self.verbose = verbose
        self.dense_retriever = dense_retriever

    def generate_rm3_query(self, qid,query, cluster_heads):
 
        batch = pd.DataFrame([[docno,score[0]] for docno, score in cluster_heads], columns=['docno',"score"])
        batch['qid'] = qid
        #batch['qid'] = [qid[0]] * len(batch)
        batch['query'] = query
 
       # expander = pt.rewrite.RM3(self.terrier_index, fb_terms=10, fb_docs=5, fb_lambda=0.4)
        expander = pt.rewrite.RM3(self.terrier_index, fb_terms=10, fb_docs=5, fb_lambda=0.3)
        expanded_query_res = expander.transform(batch)
        results = self.bm25_retriever.transform(expanded_query_res)
        results = results.sort_values('score', ascending=False)
        documents = results.docno.tolist()[:128]
        scores = results.score.tolist()[:128]
 
        return documents, scores

    def min_max_scaling(self,results):
        results = results.copy()
        results["score"] = results.groupby('qid')["score"].transform(lambda x: minmax_scale(x))
        return results
    

    def get_prioritized_docs(self,candidates, bm25_scores, tct_scores, alpha, beta, gamma,delta, cluster_neigh_lookup, cluster_rm3_lookup):
        bm25_candidates = np.array([bm25_scores.get(doc_id,0) for doc_id in candidates])
        
        tct_candidates = np.array([tct_scores.get(doc_id,0) for doc_id in candidates])

        
        if len(cluster_neigh_lookup) >0:
            cluster_neigh_scores = np.array([cluster_neigh_lookup.get(doc_id,0) for doc_id in candidates])
            cluster_rm3_scores = np.array([cluster_rm3_lookup.get(doc_id,0) for doc_id in candidates])

            # score_differences = ((alpha * bm25_candidates) + ( beta * tct_candidates)) - mean_cluster
            score_diff_features = {docno: score for docno,score in zip(candidates, cluster_neigh_scores)}
            rm3_features = {docno: score for docno,score in zip(candidates, cluster_rm3_scores)}

            estimated_scores = (alpha * bm25_candidates) + ( beta * tct_candidates) + (gamma * cluster_neigh_scores) + (delta*cluster_rm3_scores)
            return bm25_candidates,tct_candidates,score_diff_features,rm3_features,estimated_scores
        else:
            estimated_scores = (alpha * bm25_candidates) + ( beta * tct_candidates)
            return bm25_candidates,tct_candidates,estimated_scores
 
    def _drop_docnos_from_counters(self, docnos, counters):
        for docno in docnos:
            for c in counters:
                del c[docno]

    def transform(self, df1: pd.DataFrame ) -> pd.DataFrame:
        
        result = {'qid': [], 'query': [], 'docno': [], 'rank': [], 'score': [], 'iteration': []}
        df1 = self.min_max_scaling(df1)
        df1 = dict(iter(df1.groupby(by='qid')))
        qids = df1.keys()
        alpha = 0.2
        beta = 0.2
        gamma = 0.2
        delta = 0.2


        if self.verbose:
            qids = logger.pbar(qids, desc='hybrid retrieval based re-ranking', unit='query')


        for qid in qids:

            scores = {}
            lookup_cross_scores = {}
            df2 = self.dense_retriever.transform(pd.DataFrame({'qid': [qid], 'query': df1[qid]['query'].iloc[0]}))
            df2 = self.min_max_scaling(df2)
            df2 = dict(iter(df2.groupby(by='qid')))

            result1 = dict(zip(df1[qid]['docno'].values, df1[qid]['rank'].values)) # initial results {docno: rel score}
            result2= dict(zip(df2[qid]['docno'].values, df2[qid]['rank'].values))

            all_docs = set(list(result1.keys()) + list(result2.keys()))

            bm25_scores = dict(zip(df1[qid]['docno'].values, df1[qid]['score'].values)) # initial results {docno: rel score}

            tct_scores= dict(zip(df2[qid]['docno'].values, df2[qid]['score'].values))
            rr_1 = [1/(rank+60) for rank in result1.values()]
            rr_2 = [1/(rank+60) for rank in result2.values()]

            rr_result1 = dict(zip(result1.keys(), rr_1))
            rr_result2 = dict(zip(result2.keys(), rr_2))

            hybrid_scores = {docno: rr_result1.get(docno, 0) + rr_result2.get(docno, 0) for docno in all_docs}

            res_map = [Counter(hybrid_scores)]

            # result1 = Counter(dict(zip(df1[qid].docno, df1[qid].rank))) # initial results {docno: rel score}
            candidates = list(hybrid_scores.keys())
            iteration=0  
            query = df1[qid]['query'].iloc[0]
            num_batch = 0 



            while len(scores) < self.num_results:
                final_candidates = [can for can in candidates if can not in scores]
                if num_batch >0:
                    cluster_heads = [doc for doc, _ in Counter(scores).most_common(self.top_s)]

                    # Step 2: Collect neighbors and scores efficiently
                    cluster_neigh_lookup = defaultdict(list)
                    cluster_rm3_lookup = defaultdict(list)
                    rm3neighbors, rm3neighbor_scores = self.generate_rm3_query(qid,query,Counter(scores).most_common(self.top_s))
                    for neighbor, neigh_score in zip(rm3neighbors, rm3neighbor_scores):
                        cluster_rm3_lookup[neighbor].append(neigh_score)
                        

                    # Step 3: Compute mean scores
                    cluster_rm3_lookup = {key: mean(neigh_scores) for key, neigh_scores in cluster_rm3_lookup.items()}
                    for cluster_head in cluster_heads:
                        # Fetch neighbors and weights once per cluster_head
                        neighbors, laff_scores = self.laff_graph.neighbours(cluster_head, weights=True)
                    
                        
                        # Use defaultdict to accumulate scores
                        for neighbor, neigh_score in zip(neighbors, laff_scores):
                            cluster_neigh_lookup[neighbor].append( lookup_cross_scores[cluster_head][0] * neigh_score)
                        
                    # Step 3: Compute mean scores
                    cluster_neigh_lookup = {key: mean(neigh_scores) for key, neigh_scores in cluster_neigh_lookup.items()}
                    bm25_feature_dict,tct_feature_dict,score_diff_features,rm3_features,estimated_scores = self.get_prioritized_docs(final_candidates,bm25_scores,tct_scores,alpha,beta,gamma,delta, cluster_neigh_lookup,cluster_rm3_lookup)
                else:
                    bm25_feature_dict,tct_feature_dict,estimated_scores = self.get_prioritized_docs(final_candidates,bm25_scores,tct_scores,alpha,beta,None,None, [],[])
                size = min(self.batch_size, self.num_results - len(scores))
                batch_selected = sorted(zip(final_candidates,estimated_scores), key= lambda x: x[1], reverse=True)[:size]
                docnos_final = [doc[0] for doc in batch_selected]
                 # get either the batch size or remaining budget (whichever is smaller)
                if num_batch < self.cross_enc_budget:
                    batch = pd.DataFrame(docnos_final, columns=['docno'])
                    batch['qid'] = qid
                    batch['query'] = query
                    reranked_batch = self.scorer(batch)
                    unscaled_scores = list(reranked_batch["score"].values)
                    reranked_batch = self.min_max_scaling(reranked_batch)
                    
                    bm25_features = np.array([bm25_scores.get(doc_id,0) for doc_id in docnos_final])
                    tct_features = np.array([tct_scores.get(doc_id,0) for doc_id in docnos_final])
                    
                    if num_batch >0:
                        score_diff_feat = np.array([score_diff_features.get(doc_id,0) for doc_id in docnos_final])
                        rm3_feat = np.array([rm3_features.get(doc_id,0) for doc_id in docnos_final])
                        features = np.concatenate((bm25_features.reshape(-1,1),tct_features.reshape(-1,1),score_diff_feat.reshape(-1,1),rm3_feat.reshape(-1,1)),axis=1)
                        params = scipy.optimize.lsq_linear(features, list(reranked_batch["score"].values),lsq_solver="exact", bounds=(0,1))
                        alpha = params["x"][0]
                        beta = params["x"][1]
                        gamma = params["x"][2]
                        delta = params["x"][3]
                    else:
                        features = np.concatenate((bm25_features.reshape(-1,1),tct_features.reshape(-1,1)),axis=1)
                        params = scipy.optimize.lsq_linear(features, list(reranked_batch["score"].values),lsq_solver="exact", bounds=(0,1))
                        alpha = params["x"][0]
                        beta = params["x"][1]

                    batch_docnos = reranked_batch.docno

                    lookup_cross_scores.update({k: (s, iteration) for k, s in zip(reranked_batch.docno, reranked_batch.score)})
                    scores.update({k: (s, iteration) for k, s in zip(reranked_batch.docno, unscaled_scores)})
                else:
                    batch_selected = sorted(zip(final_candidates,estimated_scores), key= lambda x: x[1], reverse=True)[:self.num_results - len(scores)]
                    lookup_cross_scores.update({k: (s, iteration) for k, s in batch_selected})
                    scores.update({k: (s, iteration) for k, s in batch_selected})

                num_batch+=1

                iteration+=1   

            
            result['qid'].append(np.full(len(scores), qid))
            result['query'].append(np.full(len(scores), query))
            result['rank'].append(np.arange(len(scores)))
            for did, (score, i) in Counter(scores).most_common():
                result['docno'].append(did)
                result['score'].append(score)
                result['iteration'].append(i)   

            # Backfill unscored items
            if self.backfill and len(scores) < self.num_results:
                last_score = result['score'][-1] if result['score'] else 0.
                count = min(self.num_results - len(scores), len(res_map[0]))
                result['qid'].append(np.full(count, qid))
                result['query'].append(np.full(count, query))
                result['rank'].append(np.arange(len(scores), len(scores) + count))
                for i, (did, score) in enumerate(res_map[0].most_common()):
                    if i >= count:
                        break
                    result['docno'].append(did)
                    result['score'].append(last_score - 1 - i)
                    result['iteration'].append(-1)
    

        return pd.DataFrame({
            'qid': np.concatenate(result['qid']),
            'query': np.concatenate(result['query']),
            'docno': result['docno'],
            'rank': np.concatenate(result['rank']),
            'score': result['score'],
            'iteration': result['iteration'],
        })