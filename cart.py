import numpy as np
import pandas as pd
import pickle


with open('data/h&m/item_cd2nm_map.pickle', 'rb') as f:
    item_cd2nm_map = pickle.load(f)

def get_recc(learned_item_embeddings, item_col, positives, negatives=None, topn=5):
    rec_result = learned_item_embeddings.most_similar_cosmul(positive=positives, negative=negatives, topn=topn)
    rec_result_df = pd.DataFrame(rec_result, columns=[item_col, 'score'])
    rec_result_df['item_nm'] = rec_result_df[item_col].map(item_cd2nm_map)
    rec_result_df['rank'] = range(1, topn+1)
    return rec_result_df

def get_user_vector(emb, item_seq, method='sum'):
    """
    1. Lookup embeddings from given code
    2. Sum all embeddings in a sequence
    3. return summed embeddings

    parameters
    -----------------
    emb : KeyedVectors, trained embeddings using Word2Vec
    item_seq : list of string, list of item id within a seq
    """
    cold_items =[]
    item_emb_lst = []
    for item in item_seq:
        try:
            item_emb = emb[item]
            item_emb_lst.append(item_emb)
        except KeyError:
            cold_items.append(item)
    # if all cold items
    if len(item_seq) == len(cold_items):
        print("all cold items, return zero vector")
        return np.array([0]), cold_items
    item_emb_arr = np.array(item_emb_lst)
    if len(item_emb_lst) == 1:
        return item_emb_lst[0], cold_items
    if method == 'sum':
        user_emb = np.sum(item_emb_arr, axis=0)
    return user_emb, cold_items