from elasticsearch import Elasticsearch
import csv
import time
from typing import List, Dict


ES_URI = "https://hackathon-deployment.es.us-east1.gcp.elastic-cloud.com:9243"
ES_USER = "elastic"
ES_PASSWORD = "xjzoHBeQccXmR83VqwjOFqcH"
ES_TIMEOUT = 600


def __main__():
    # create an Elasticsearch client instance
    es = Elasticsearch(
        [ES_URI],
        basic_auth=(ES_USER, ES_PASSWORD),
        request_timeout=ES_TIMEOUT,
    )

    # read csv file
    with open("swu_extractions.csv", "r") as f:
        reader = csv.DictReader(f)
        swu_extracions = list(reader)

    # create a list of dictionary to group the keywords by their ids
    def group_keywords_by_tweet_id(splitted_map: List[Dict]) -> List[Dict]:
        grouped_map = {}
        for elem in splitted_map:
            if elem["ids"] not in grouped_map:
                grouped_map[elem["ids"]] = []
            grouped_map[elem["ids"]].append(elem["keywords"])

        # concatenate the data field of each group into a single string
        consolidated_map = [
            {"id": k, "keywords": " ".join(v)} for k, v in grouped_map.items()
        ]
        return consolidated_map

    id_to_keywords_sentence = group_keywords_by_tweet_id(swu_extracions)

    # # search all
    # search_all_query = {"query": {"match_all": {}}}
    # search_all_results = es.search(
    #     index="nato_hackathon_disinfo", query={"match_all": {}}, size=408
    # )

    # index_to_ids = [
    #     {"index": i, "id": j["_id"]}
    #     for i, j in enumerate(search_all_results["hits"]["hits"])
    # ]

    # with open("index_to_ids.csv", "w") as f:
    #     writer = csv.DictWriter(f, fieldnames=["index", "id"])
    #     writer.writeheader()
    #     writer.writerows(index_to_ids)

    # Exemple of a search query looking for the keywords "kyiv" or "knife" in the headlines and articles fields
    # define the search query
    search_query = {
        "query": {
            "bool": {
                "must": [
                    {
                        "multi_match": {
                            "query": "kyiv knife",
                            "fields": ["headlines", "articles"],
                        }
                    }
                ]
            }
        }
    }

    # def build_query(keywords: str) -> dict:
    #     return {
    #         "query": {
    #             "bool": {
    #                 "must": [
    #                     {
    #                         "multi_match": {
    #                             "query": keywords,
    #                             "fields": ["headlines", "articles"],
    #                         }
    #                     }
    #                 ]
    #             }
    #         },
    #         "size": 408,
    #     }
    def build_query(keywords: str) -> dict:
        return {
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": keywords,
                                "fields": ["headlines"],
                            }
                        }
                    ]
                }
            },
            "size": 408,
    }
    # build the query and return the articles
    def get_articles_ids(keywords: str) -> List[Dict]:
        search_query = build_query(keywords)
        search_results = es.search(index="nato_hackathon_disinfo", body=search_query)
        return [k["_id"] for k in search_results["hits"]["hits"]]

    # get the articles for each tweet
    time_start = time.time()
    all_tweets = []
    for i, tweet in enumerate(id_to_keywords_sentence):
        if i==100:
            break
        if i % 10 == 0:
            print(f"Processed {i} tweets in {time.time() - time_start} seconds")
        tweet["articles"] = get_articles_ids(tweet["keywords"])
        all_tweets.append(tweet)

    with open("tweets_with_articles.csv", "w") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "keywords", "articles"])
        writer.writeheader()
        writer.writerows(all_tweets)
    print(f"Processed all tweets in {time.time() - time_start} seconds")


if __name__ == "__main__":
    __main__()
