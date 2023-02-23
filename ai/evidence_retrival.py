import numpy as np
import pandas as pd
import spacy
import pickle
import torch
from sentence_transformers import SentenceTransformer, util

nlp = spacy.load('en_core_web_sm', disable = ['ner', 'parser', 'tagger','lemmatizer','attribute_ruler' ])
nlp.max_length = 1500000
nlp.add_pipe(nlp.create_pipe('sentencizer'))

model = SentenceTransformer('all-MiniLM-L6-v2')

"""Explore to use SBERT (1st stage text ranker) evidence retriever

split doc text into paragraphs / sentences -> Obtain top K paragraph / sentences that semantically similar to claim text -> output doc text prefiltered and sorted dataset

- text are normalised
- invalid samples are filtered

output is consistent with original dataset structure to be used for subsequent embedding and model training
"""
from typing import List
import os
import logging
import nltk
nltk.download("punkt")
import torch
import pandas as pd
from tqdm import tqdm
# from text_utils import simple_text_norm
from sentence_transformers import SentenceTransformer, util

# logger = logging.getLogger(__name__)
# logging.basicConfig(level=logging.INFO,
#                     filename=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
#                                           "evidence_retrieval_with_sbert.log"),
#                     format='%(asctime)s [%(threadName)-12.12s] - %(name)s - %(levelname)s - %(message)s',
#                     datefmt='%Y-%m-%d %H:%M:%S')
# # set up logging to console
# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# # set a format which is simpler for console use
# formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')
# console.setFormatter(formatter)
# logging.getLogger(__name__).addHandler(console)

device = "cuda" if torch.cuda.is_available() else "cpu"

# all-MiniLM-L6-v2: dim: 384, max token size: 256
# 'multi-qa-mpnet-base-dot-v1': dim: 768, max token size: 512
sbert_model_name = 'multi-qa-mpnet-base-dot-v1'
sbert_model = SentenceTransformer(sbert_model_name)  # dim: 768
SIMPLIFIED_MODEL_NAME = "sbertqa"
sbert_model.to(device)
sbert_model.eval()
# logger.info("%s model loaded", sbert_model_name)

def generated_matched_dataset(ground_truth_path, tweets_path):
    data_compare = pd.read_csv(ground_truth_path, encoding='utf-8')
    tweets = pd.read_csv(tweets_path, encoding='utf-8')
    er_mid = []
    er_mid = pd.DataFrame(er_mid)
    er_mid['keywords'] = tweets['keywords']
    articls = []
    headlins = []
    for i in range(len(tweets)):
        headlinestext = ''
        articletext = ''
        for article in tweets.iloc[i]['articles'][1:-1].split(","):
            if article == '':
                articletext += article
            else:
                headlinestext+= data_compare.loc[data_compare['id'] == article.strip()[1:-1], 'headlines'].iloc[0]+"\n "
                articletext+= data_compare.loc[data_compare['id'] == article.strip()[1:-1], 'content'].iloc[0]+"\n\n "
        headlins.append(headlinestext)
        articls.append(articletext)
    er_mid['headlines'] = headlins
    er_mid['content'] = articls
    er_mid.to_csv("er_processed.csv")

def get_top_k_paragraph(claim_text: str, doc_text: str, k=5) -> List[str]:
    if type(doc_text) == float:
        doc_paragraphs = ['']
    else:
        doc_paragraphs = doc_text.split("\n")
    norm_doc_paragraphs = [doc_paragraph for doc_paragraph in doc_paragraphs]
    # print("doc_paragraphs size: ", len(doc_paragraphs))
    if len(norm_doc_paragraphs) == 1:
        return norm_doc_paragraphs

    return get_top_k(claim_text, norm_doc_paragraphs, k=k)


def get_top_k_sentences(claim_text: str, doc_text: str, k=5) -> List[str]:
    if type(doc_text) == float:
        sentences = ''
    else:
        sentences = nltk.sent_tokenize(doc_text)
    norm_sentences = [sent for sent in sentences]
    MIN_SIZE_THREHSOLD = 2
    # skip evidence re-ranking if sentence size is below a certain threshold
    if len(norm_sentences) <= MIN_SIZE_THREHSOLD:
        return norm_sentences
    return get_top_k(claim_text, norm_sentences, k=k)


def get_top_k(claim_text: str, evidence_text_list: List[str], k=5) -> List[str]:
    """
    return top K semantic similar evidence text list
    :param claim_text:
    :param evidence_text_list: evidence text list to compare with claim text
    :param k:
    :return:
    """
    if len(evidence_text_list) == 1:
        return evidence_text_list

    claim_text_embedding = sbert_model.encode(claim_text, convert_to_tensor=True,
                                              normalize_embeddings=False)  # torch.Size([384])
    evidences_embebddings = sbert_model.encode(evidence_text_list, convert_to_tensor=True,
                                               normalize_embeddings=False)  # torch.Size([n, 384])
    # cosine similarity with efficient dot product and normalisation already applied in util.semantic_search()
    top_k_sem_search_results = util.semantic_search(claim_text_embedding, evidences_embebddings, top_k=k)
    # print("sem_search_results: ", top_k_sem_search_results) # [[{'corpus_id': 19, 'score': 0.4779782295227051}, {'corpus_id': 15, 'score': 0.23351801931858063} ...]]

    top_k_evidences = [evidence_text_list[evid_res["corpus_id"]] for evid_res in top_k_sem_search_results[0]]
    # print("claim_text: ", claim_text)
    # print("top_k_evidences: ", top_k_evidences)
    return top_k_evidences


def generate_evidence_filtered_factify2_dataset(dataset_path, split_type, top_k, enable_filtering):
    """
    give a factify2 dataset, we generate evidence filtered dataset with vary length (i.e., top K)
    :param dataset_path: train.csv, val.csv, test.csv
    :param split_type: "paragraph" | "sentence"
    :param top_k:
    :param enable_filtering: filtering invalid samples (not to apply to test set)
    :return:
    """
    # logger.info("loading dataset from [%s] ... " % dataset_path)
    data_df = pd.read_csv(dataset_path, sep=',', encoding="UTF-8", index_col=[0], header=0)
    # logger.info("[%s] data loaded. " % data_df.size)
    evidence_filtered_df = data_df.copy()
    dropped_samples = 0
    # logger.info("processing factify2 evidence re-ranked dataset with 'split_type': [%s], "
    #             "'TOP_K': [%s], 'enable_invalid_sample_filtering': [%s]", split_type, top_k,
    #             enable_filtering)
    for row in data_df.iterrows():
        id = int(row[0])
        claim_text = row[1]["keywords"]
        doc_text = row[1]["content"]
        top_evidences = []

        if split_type == "paragraph":
            top_evidences = get_top_k_paragraph(claim_text, doc_text, k=top_k)
        elif split_type == "sentence":
            top_evidences = get_top_k_sentences(claim_text, doc_text, k=top_k)
        else:
            raise ValueError("Unsupported doc text split type [%s]!" % split_type)
        refined_doc_text = " . ".join(top_evidences)

        # make changes to content
        evidence_filtered_df["keywords"][evidence_filtered_df.index == id] = claim_text
        evidence_filtered_df["content"][evidence_filtered_df.index == id] = refined_doc_text
    evidence_filtered_df.to_csv(dataset_path.replace(".csv", "_{}_{}_top_{}.csv".format(SIMPLIFIED_MODEL_NAME,
                                                                                        split_type, top_k)),
                                encoding="utf-8", sep="\t")
    # logger.info("all [%s] records exported.", evidence_filtered_df.size)
    # logger.info("total [%s] samples dropped.", dropped_samples)


# if __name__ == '__main__':
#     # manually change 'split_type' and "TOP_K" to generate vary experiment settings
#     # options for top sentences: 5, 10, 15, 20, 25
#     # options for top paragraphs: 5
#     # data_set_path = "/home/jerry/data/factify2/factify2/val.csv"
#     # data_set_path = "C:\\data\\factchecking\\de-factify\\factify2\\val.csv"
#     # TOP_K = 10
#     # split_type = "sentence"  # option: "paragraph" | "sentence"
#     # only enable data filtering for train and val. DO NOT apply to test test!
#     # enable_invalid_sample_filtering = True

#     # configure all experiments as follows
#     evidence_filtering_settings = [("/home/jerry/data/factify2/factify2/train.csv", "paragraph", 5, True),
#                                    ("/home/jerry/data/factify2/factify2/val.csv", "paragraph", 5, True),
#                                    ("/home/jerry/data/factify2/factify2test/test.csv", "paragraph", 5, False),

#                                    ("/home/jerry/data/factify2/factify2/train.csv", "sentence", 5, True),
#                                    ("/home/jerry/data/factify2/factify2/val.csv", "sentence", 5, True),
#                                    ("/home/jerry/data/factify2/factify2test/test.csv", "sentence", 5, False),

#                                    ("/home/jerry/data/factify2/factify2/train.csv", "sentence", 10, True),
#                                    ("/home/jerry/data/factify2/factify2/val.csv", "sentence", 10, True),
#                                    ("/home/jerry/data/factify2/factify2test/test.csv", "sentence", 10, False),

#                                    ("/home/jerry/data/factify2/factify2/train.csv", "sentence", 15, True),
#                                    ("/home/jerry/data/factify2/factify2/val.csv", "sentence", 15, True),
#                                    ("/home/jerry/data/factify2/factify2test/test.csv", "sentence", 15, False),

#                                    ("/home/jerry/data/factify2/factify2/train.csv", "sentence", 25, True),
#                                    ("/home/jerry/data/factify2/factify2/val.csv", "sentence", 25, True),
#                                    ("/home/jerry/data/factify2/factify2test/test.csv", "sentence", 25, False)
#                                    ]
#     for exp_i_data_path, exp_i_split_type, exp_i_top_k, exp_i_enable_filtering in tqdm(evidence_filtering_settings):
#         generate_evidence_filtered_factify2_dataset(exp_i_data_path, exp_i_split_type, exp_i_top_k,
#                                                     exp_i_enable_filtering)




generate_evidence_filtered_factify2_dataset("er_processed.csv", "sentence", 5, False)