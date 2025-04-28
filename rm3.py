import pyterrier as pt
import pyterrier_alpha as pta
from pyterrier.measures import *
from pyterrier_t5 import MonoT5ReRanker
import pandas as pd
# Initialize PyTerrier
if not pt.started():
    pt.init()


dataset = pt.get_dataset('irds:msmarco-passage-v2')
scorer = pt.text.get_text(dataset, 'text') >> MonoT5ReRanker(verbose=False, batch_size=16)


indexref  = "/home/rathee/.pyterrier/artifacts/243cf93b791d03a9c208d1a7756595ecfc0bad2c7598d87266ca8760d309cfe2"
#indexref  = "/home/rathee/llmgar/indices/bm25_msmarco_passage"
existing_index = pt.IndexFactory.of(indexref)
bm25_retriever = pt.terrier.Retriever(existing_index, wmodel="BM25")


# Perform the first retrieval to get top-k documents
initial_results = bm25_retriever.search("information retrieval")

# Now, apply RM3-like expansion
# Here you manually extract top terms from the top-ranked documents
# For simplicity, we use PyTerrier's built-in functionality to perform RM3-style expansion

# Extract feedback documents (e.g., top 10)
feedback_docs = initial_results.head(10)
# Use PyTerrier to expand the query based on the feedback documents

expander = pt.rewrite.RM3(existing_index, fb_docs=10, fb_terms=10)

expanded_query_res = expander.transform(feedback_docs)


new_query = expanded_query_res['query'][0]

print(expanded_query_res)
print(new_query)

#dataset = pt.get_dataset(f'irds:msmarco-passage/trec-dl-2019/judged')
dataset = pt.get_dataset(f'irds:msmarco-passage-v2/trec-dl-2022/judged')


qrels = pd.read_csv(f"/home/rathee/llmgar/dl.dedup.qrels/dl22.dedup.qrels", header=None, sep=" ", names=["qid", "Q0", "docno", "label"],
                    dtype={"qid":str, "Q0":str, "docno":str, "label":int})


bm25 = pt.terrier.Retriever(existing_index, wmodel="BM25")
c = 100


rm3_pipe = bm25 % c >> pt.rewrite.RM3(existing_index, fb_terms=10, fb_docs=10, fb_lambda=0.6) >> bm25 % c 

# result = pt.Experiment([bm25 % c  , rm3_pipe % c >> pt.rewrite.reset() >> scorer],
#             dataset.get_topics(),
#             dataset.get_qrels(),
#             [nDCG@10, nDCG@c, R(rel=2)@c],
#             names=["BM25>>MonoT5", "BM25(RM3)>>MonoT5"]
#             )


result = pt.Experiment([bm25 % c  , rm3_pipe % c >> pt.rewrite.reset() >> scorer],
            dataset.get_topics(),
            qrels,
            [nDCG@10, nDCG@c, R(rel=2)@c],
            names=["BM25>>MonoT5", "BM25(RM3)>>MonoT5"]
            )


print(result.T)