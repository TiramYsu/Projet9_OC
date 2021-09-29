import logging

import azure.functions as func
import pickle
import surprise
import pandas as pd
import numpy as np
import json
from numpyencoder import NumpyEncoder
5
pkl_filename = './Prediction/pickle_surprise_model_KNNWithMeans.pkl'

# Load from file
with open(pkl_filename, 'rb') as file:
    pickle_model = pickle.load(file)

articles_df = pd.read_csv('./Prediction/articles_metadata.csv')

def predict_best_category_for_user(user_id, model, articles_df):
    predictions = {}
    
    #Category 1 to 460
    for i in range(1, 460):
        _, cat_id, _, est, err = model.predict(user_id, i)
        
        #Keep prediction only if we could keep it.
        if (err != True):
            predictions[cat_id] = est
    
    best_cats_to_recommend = dict(sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:5])
    
    recommended_articles = []
    for key, _ in best_cats_to_recommend.items():
        recommended_articles.append(int(articles_df[articles_df['category_id'] == key]['article_id'].sample(1).values))
    
    #return random_articles_for_best_cat, best_cat_to_recommend
    return recommended_articles, best_cats_to_recommend

def main(req: func.HttpRequest) -> func.HttpResponse:
    logging.info('Python HTTP trigger function processed a request.')

    name = req.params.get('userId')
    if not name:
        try:
            req_body = req.get_json()
        except ValueError:
            pass
        else:
            name = req_body.get('userId')

    if name:
        
        results, _ = predict_best_category_for_user(int(name), pickle_model, articles_df)
        json_result = json.dumps(results, cls=NumpyEncoder)
        return func.HttpResponse(json_result)
    else:
        return func.HttpResponse(
             "This HTTP triggered function executed successfully. Pass a name in the query string or in the request body for a personalized response.",
             status_code=200
        )