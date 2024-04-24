def HR_at_K(y_true,y_pred,k=2):
    n_samples = len(y_true)
    correct = 0
    for i in range(n_samples):
        top_k_index = y_pred[i][:k]
        if y_true[i] in top_k_index:
            correct +=1
    return round(correct/n_samples, 3)

def average_precision(ranked_list, positive_items,k=3):  
    ranked_list = ranked_list[:k]
    positive_items = positive_items[:k]
    is_relevant = [elem in positive_items for elem in ranked_list]  
      
    cumulative_hits = 0  
    sum_precs = 0  
    for idx, correct in enumerate(is_relevant, 1):  # Note: idx starts from 1 here  
        if correct:  
            cumulative_hits += 1  
            precision = cumulative_hits / idx  #1/1 
            sum_precs += precision  
  
    if not is_relevant:  # no relevant items in list  
        return 0  
    if sum(is_relevant)==0:
        average_precision=0
    else:
        average_precision = sum_precs / sum(is_relevant)#len(is_relevant)  
    return round(average_precision  ,3)
  
def Mean_Average_Precision(y_true, y_pred, k=2):
    n_samples = len(y_true)  
    map_score = 0  
    for user in range(n_samples):  
        true_items = y_true[user] #4.0
        true_items = [int(true_items)]
        recommended_items = list(y_pred[user])  #list
        ap = average_precision(recommended_items, true_items,k)  #1
        map_score += ap  
    map_score /= n_samples  
    return round(map_score,3)


def Mean_Reciprocal_Rank(y_true, y_pred, k=2):  
    n_samples = len(y_true)  
    total_reciprocal_rank = 0  
  
    for i in range(n_samples):  
        true_labels =int( y_true[i]  )
        pred_labels = list(y_pred[i][:k])

        if true_labels in pred_labels:  
            reciprocal_rank = 1 / (pred_labels.index(true_labels) + 1)  
            total_reciprocal_rank += reciprocal_rank  
    mean_reciprocal_rank = total_reciprocal_rank / n_samples if n_samples > 0 else 0  
    return round(mean_reciprocal_rank,3)

import numpy as np
def Normalized_Discounted_Cumulative_Gain(y_true, y_pred, k=2):  
    n_samples = len(y_true)  
    NDCG_sum=0
    for i in range(n_samples):  
        true_labels =int( y_true[i]  )
        pred_labels = list(y_pred[i][:k])
        
        if true_labels in pred_labels:  
            rank_idx =  pred_labels.index(true_labels) + 1
            DCG=1/(np.log2(rank_idx+1))
        else:
            DCG=0
        IDCG= 1
        NDCG = DCG  
        NDCG_sum+=NDCG
    NDCG_average=NDCG_sum/n_samples
    return round(NDCG_average,3)

