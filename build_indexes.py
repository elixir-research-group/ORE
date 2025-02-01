import torch
from pyterrier_dr import FlexIndex, TasB, TctColBert, NumpyIndex
import pyterrier as pt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



### to create tct Numpy index

dataset = pt.get_dataset('irds:msmarco-passage')

model = TctColBert('castorini/tct_colbert-v2-hnp-msmarco')
index_pipeline = TctColBert('castorini/tct_colbert-v2-hnp-msmarco') >> NumpyIndex('indices/castorini_tct_colbert-v2-hnp-msmarco')
index_pipeline.index(dataset.get_corpus_iter())


### to create tasb flex index


model = TasB.dot(batch_size=1, device=device)
index = FlexIndex('indices/msmarco-passage.tasb.flex')
idx_pipeline = model >> index
idx_pipeline.index(pt.get_dataset('irds:msmarco-passage').get_corpus_iter())