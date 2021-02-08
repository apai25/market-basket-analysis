import numpy as np 
import pandas as pd 

dataset = pd.read_csv('Market_Basket_Optimization.csv', header=None)
dataset = dataset.values.astype(str).tolist()

from apyori import apriori
rules = apriori(dataset, min_support=.005, min_confidence=.25, min_lift=2.7, min_length=2, max_length=2, random_state=0)
results = list(rules)

base_item = [list(result[0])[0] for result in results]
added_item = [list(result[0])[1] for result in results]
support = [result[1] for result in results]
confidence = [result[2][0][2] for result in results]
lift = [result[2][0][3] for result in results]
final_results = list(zip(base_item, added_item, support, confidence, lift))

results_in_dataframe = pd.DataFrame(final_results, columns=['Base Item', 'Added Item', 'Support', 'Confidence', 'Lift'])
results_in_dataframe.to_csv('results.csv')