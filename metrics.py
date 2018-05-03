import math
from sklearn.metrics import roc_auc_score
import numpy as np

def dcg_(l, k=10):
    dcg_v=0
    for i in range(k):
        dcg_v=dcg_v+l[i]/math.log(i+2,2)
    return dcg_v

# hit ratio metric
def hr(predicted_scores):
    
    # set a threshold 0f 0.5 to set a hit (1) or a miss (0) for a user.
    predicted_scores = sorted(predicted_scores, reverse=True)
    y_pred = np.zeros(len(predicted_scores))
    threshold_hits = np.array(predicted_scores) >= 0.5
    y_pred[threshold_hits] = 1
    
    return np.mean(y_pred)

# Calculate the predictions
def predict(model, uid, pids):

    item_matrix = model.get_layer('item_embedding').get_weights()[0][pids]
    user_vector = model.get_layer('user_embedding').get_weights()[0][uid]

    scores_G = (np.dot(user_vector,
                     item_matrix.T))
    return scores_G

def full_auc(model, ground_truth):
    """
    Measure AUC, NDCG, HR for model and ground truth on all items.
    Returns:
    - float AUC
    - float NDCG
    - float HR
    """
    # testting matrix
    ground_truth = ground_truth.tocsr()
    
    # total number of users and items in the data
    no_users, no_items = ground_truth.shape
    
    # Return evenly spaced values within a given interval (0, number_of_items).
    pid_array = np.arange(no_items, dtype=np.int32)
    
    # AUC scores
    scores = []
    
    # actual scores
    
    # iterate over all the users for predictions
    for user_id, row in enumerate(ground_truth):
        # predict the model outputs
        predictions = predict(model, user_id, pid_array)
        
        # create a array consisting of 1's at the non-zero rating indexes
        true_pids = row.indices[row.data == 1]
        grnd = np.zeros(no_items, dtype=np.int32)
        grnd[true_pids] = 1
        
        # calculate the score
        if len(true_pids):
            scores.append(roc_auc_score(grnd, predictions))
           
    # calculate the ndcg value
    ndcg = dcg_(scores)/dcg_(sorted(scores, reverse=True))
            
    # calculate the hit ratio for all the scores
    hr_value = hr(scores)
    
    return (sum(scores) / len(scores)), ndcg, hr_value