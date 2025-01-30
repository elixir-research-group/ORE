from typing import Optional
import numpy as np
from collections import Counter
import pyterrier as pt
import pandas as pd
import ir_datasets
logger = ir_datasets.log.easy()

from sklearn.preprocessing import minmax_scale





class HybridCC(pt.Transformer):

    def __init__(self,
        scorer: pt.Transformer,
        num_results: int = 100,
        cross_enc_budget: int = 7,
        df2: pd.DataFrame = None,
        batch_size: Optional[int] = None,
        backfill: bool = True,
        verbose: bool = True):

        self.scorer = scorer
        self.num_results = num_results
        self.cross_enc_budget = cross_enc_budget
        self.df2 = df2
        if batch_size is None:
            batch_size = scorer.batch_size if hasattr(scorer, 'batch_size') else 16
        self.batch_size = batch_size
        self.backfill = backfill
        self.verbose = verbose



    def min_max_scaling(self,results):
        results = results.copy()
        results["score"] = results.groupby('qid')["score"].transform(lambda x: minmax_scale(x))
        return results
    

    def transform(self, df1: pd.DataFrame ) -> pd.DataFrame:
        
        result = {'qid': [], 'query': [], 'docno': [], 'rank': [], 'score': [], 'iteration': []}

        df1 = self.min_max_scaling(df1)    
        df2 = self.min_max_scaling(self.df2)

        df1 = dict(iter(df1.groupby(by='qid')))
        df2 = dict(iter(self.df2.groupby(by='qid')))

        qids = df1.keys()


        if self.verbose:
            qids = logger.pbar(qids, desc='hybrid retrieval based re-ranking', unit='query')



        for qid in qids:


            scores = {}

            result1 = dict(zip(df1[qid]['docno'].values, df1[qid]['score'].values)) 
            result2= dict(zip(df2[qid]['docno'].values, df2[qid]['score'].values))

            all_docs = set(list(result1.keys()) + list(result2.keys()))


            hybrid_scores = {docno: 0.5 * result1.get(docno, 0) + 0.5 * result2.get(docno, 0) for docno in all_docs}

            res_map = [Counter(hybrid_scores)]


            iteration=0  
            query = df1[qid]['query'].iloc[0]
            num_batch = 0 

            while len(scores) < self.num_results and num_batch < self.cross_enc_budget:

                this_res = res_map[0]

                size = min(self.batch_size, self.num_results - len(scores))

                batch = this_res.most_common(size)

                batch = pd.DataFrame(batch, columns=['docno', 'score'])
                batch['qid'] = qid
                #batch['qid'] = [qid[0]] * len(batch)
                batch['query'] = query
                    

                # go score the batch of document with the re-ranker
                batch = self.scorer(batch)
                num_batch+=1

                scores.update({k: (s, iteration) for k, s in zip(batch.docno, batch.score)})
                self._drop_docnos_from_counters(batch.docno, res_map)


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

    def _drop_docnos_from_counters(self, docnos, counters):
        for docno in docnos:
            for c in counters:
                del c[docno]