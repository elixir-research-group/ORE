from collections import Counter
from typing import List

import pandas as pd
import pyterrier as pt
import pyterrier_alpha as pta
from pyterrier_adaptive import CorpusGraph


class Cerberus(pt.Transformer):
    def __init__(self, scorer: pt.Transformer, graph: CorpusGraph, *, budget: int = 1000, verbose: bool = False):
        self.scorer = scorer
        self.graph = graph
        self.budget = budget
        self.verbose = verbose

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        result_builder = pta.DataFrameBuilder(['qid', 'query', 'docno', 'score', 'rank'])
        groups = list(inp.groupby('query'))
        for i, (query, initial_results) in enumerate(groups):
            qid = initial_results['qid'].iloc[0]
            if self.verbose:
                print(f'\nStarting Query: {query} ({i}/{len(groups)})')
            initial_results = initial_results.sort_values('score', ascending=False)
            arms = [Arm(initial_results['docno'].tolist(), name='initial_results')]
            results = {}
            while len(arms) > 0 and len(results) < self.budget:
                arm = max(arms, key=lambda x: x.estimate_utility())
                docno = arm.pull()
                if docno not in results:
                    score = self.scorer([{
                        'qid': qid,
                        'query': query,
                        'docno': docno
                    }])['score'].iloc[0]
                    neighbor_arm = Arm(self.graph.neighbours(docno), name=f'neighbors_{docno}')
                    neighbor_arm.push(score)
                    arms.append(neighbor_arm)
                    results[docno] = score
                    already_scored = ''
                else:
                    score = results[docno]
                    already_scored = '\talready_scored'
                if self.verbose:
                    print(f' pulled {docno}\tscore={score:.4f}\tarm={arm.name}\tutility={arm.estimate_utility():.4f}{already_scored}')
                arm.push(score)
                arms = [a for a in arms if not a.is_exhausted()]
                for rank, (docno, score) in enumerate(Counter(results).most_common()):
                    result_builder.extend({
                        'qid': qid,
                        'query': query,
                        'docno': docno,
                        'score': score,
                        'rank': rank,
                    })
        return result_builder.to_df()


class Arm:
    def __init__(self, docnos: List[str], name: str = ''):
        self.docnos = docnos
        self.scores = []
        self.name = name

    def is_exhausted(self):
        return len(self.docnos) == 0

    def pull(self):
        assert not self.is_exhausted()
        next_docno, self.docnos = self.docnos[0], self.docnos[1:]
        return next_docno

    def push(self, score: float):
        self.scores.append(score)

    def estimate_utility(self):
        # for now: mean of scores
        # TODO: something fancy
        if len(self.scores) == 0:
            return float('-inf')
        return sum(self.scores) / len(self.scores)
