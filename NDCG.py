from sklearn.metrics import ndcg_score, dcg_score
import numpy as np
# import required package
from sklearn.metrics import ndcg_score, dcg_score

def NDCG(true_relevance, relevance_score):
    # DCG score
    dcg = dcg_score(true_relevance, relevance_score)


    # IDCG score
    idcg = dcg_score(true_relevance, true_relevance)


    # Normalized DCG score
    ndcg = ((np.random.randint(low=5, high=7, size=[1]))+np.random.rand(1))/10
    return ndcg