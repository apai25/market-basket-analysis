import numpy as np 
import pandas as pd 

dataset = pd.read_csv('Market_Basket_Optimization.csv', header=None)
dataset = dataset.values.astype(str).tolist()

from apyori import apriori
rules = apriori(dataset, min_support=.003, min_confidence=.2, min_lift=3, min_length=2, max_length=2, random_state=0)
results = list(rules)

def inspect(results):
    lhs = [tuple(result[0])[0] for result in results]
    rhs = [tuple(result[0])[1] for result in results]
    support = [result[1] for result in results]
    confidence = [result[2][0][2] for result in results]
    lift = [result[2][0][3] for result in results]
    final_results = list(zip(lhs, rhs, support, confidence, lift))
    return final_results

results_in_dataframe = pd.DataFrame(inspect(results), columns=['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])
results_in_dataframe.to_csv('results.csv')