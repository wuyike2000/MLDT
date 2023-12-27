import os
from sentence_transformers import SentenceTransformer, util
import torch
import torch.nn.functional as F

class Similarity:
    def __init__(self):
        # Load model
        self.model = SentenceTransformer('../../pretrain/all-MiniLM-L6-v2',device='cuda')

    def sim_compute(self, query, demo):
        embedding1 = self.model.encode(query, show_progress_bar=False,device='cuda',convert_to_tensor=True)
        embedding2 = self.model.encode(demo,batch_size=8192,show_progress_bar=False, device='cuda',convert_to_tensor=True)
        cosine_scores = util.pytorch_cos_sim(embedding1, embedding2)[0]
        return cosine_scores


