import os
import requests
from fastapi import FastAPI
import sys
from google.cloud import storage
import string
import random
import string 
import pandas as pd 
from preprocess import generate_preprocess_dataset
from elastic_searc import get_tweets_with_articles
from evidence_retrival import generated_matched_dataset, generate_evidence_filtered_hackaton_dataset
from misinfo_entailment import get_entailment
app = FastAPI()

print("environement", os.environ)
INPUT_PATH = "inputs"
OUTPUT_PATH = "outputs"
print("temp folder : ", INPUT_PATH)
if not os.path.exists(INPUT_PATH):
    os.makedirs(INPUT_PATH)

if not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)


@app.post("/hackaton" )
async def solution(data:dict):
    print(data)
    filename = os.path.join(INPUT_PATH, data["filename"])
    base = os.path.splitext(os.path.basename(filename))[0]
    # process csv file tweet and extract keyword
    print("processing and extracting keyword from tweet")
    df_tweets_processed, df_tweets_extractions = generate_preprocess_dataset(filename, OUTPUT_PATH) # base + "_extractions.csv"
    filename_extraction = os.path.join(OUTPUT_PATH, os.path.basename(filename.replace(".csv", "_extractions.csv")))
    # get article compatible by keyword with tweets 
    print("match article - tweet by keyword & elastic search")
    match_path = os.path.join(OUTPUT_PATH, base + "_tweets_with_articles.csv")
    df_tweets_with_articles = get_tweets_with_articles(filename_extraction, match_path) # pd.read_csv("tweets_with_articles.csv") #

    # apply evidence retrieval
    print("apply evidence retrival")
    filename_er = os.path.join(OUTPUT_PATH, base )+ "_er_processed.csv"
    er_mid = generated_matched_dataset(ground_truth_path="groundtruth_concat.csv", 
                                        tweets_path="tweets_with_articles.csv",
                                        tweets_data=df_tweets_processed)
    er_mid.to_csv(filename_er)

    evidence_filtered_df, filename_evidence = generate_evidence_filtered_hackaton_dataset(filename_er, "sentence", 3, False)

    # apply entailment
    print("apply entailment")

    result = get_entailment(filename_evidence)

    print(result)
    return {"output":""}

if __name__ == '__main__':
    import uvicorn
    #app.run(host='0.0.0.0', port=6004)
    uvicorn.run(app, host='0.0.0.0', port=6004)