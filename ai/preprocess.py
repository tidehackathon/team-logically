import os
import json
import pandas as pd
import nltk
import numpy as np
from gensim.utils import deaccent
from unidecode import unidecode
import csv 
import sys
import re
import nltk
import string
from typing import List, Tuple
from gensim.utils import deaccent

import itertools
import subprocess
from nltk.corpus import stopwords
from nltk.tokenize import TweetTokenizer

from collections import Counter
import csv
import codecs
from tqdm import tqdm 
from rake_nltk import Rake
import pickle
import string

try:
    from emoji import demojize, get_emoji_regexp
except ImportError as error:
    install("emoji")
    from emoji import demojize, get_emoji_regexp
    

nltk_tweet_tokenizer = TweetTokenizer(strip_handles=False, reduce_len=True)
punctuation_filter = lambda t: filter(lambda a: a not in string.punctuation, t)
stop_words_filter = lambda t: filter(lambda a: a not in _stop_words, t)
r = Rake()
def preprocessing(input_text):
    """normalise text and extract text from double quotes
    claims within double quotes are commonly seen in politifact claims
    Note: need consider not to make 'double_quotes_extraction' as default since claims usually include who(claimer/speaker) make the claim
        which is essential info,
        e.g., https://www.politifact.com/factchecks/2023/jan/09/donald-trump/fact-checking-donald-trump-drug-overdose-rates-his/
    :return normalised text
    """
    # Removing quotation marks front and back.
    # Some claims have quotation marks in front and back of string.
    norm_text = deaccent(input_text)
    norm_text = unidecode(norm_text)
    return norm_text


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    
def give_emoji_free_text(text):
    if isinstance(text, bytes):
        text = text.decode('utf8')
    return get_emoji_regexp().sub(r'', text)


def if_remove_too_long_emojis(decoded_tweet):
    """workdaround to avoid too many emojis causing BERT out of index error"""
    from emoji import get_emoji_regexp
    count_emoji = len(re.findall(get_emoji_regexp(), decoded_tweet))

    MAX_EMOJI_TRUNCATE_THRESHOLD = 8
    return count_emoji >= MAX_EMOJI_TRUNCATE_THRESHOLD


def emoji2text(tweet_text):
    # transcode the UTF-16 surrogate pair for emoji
    tweet = tweet_text.encode('utf-16', 'surrogatepass').decode('utf-16')

    if_truncate = if_remove_too_long_emojis(tweet)
    # tokenizer may separate consecutive emoji UTF-16 characters, thus, apply demojize can maximally preserve emoji
    # e.g., US flag emoji '\ud83c\uddfa\ud83c\uddf8' which should be converted to united_states
    #     if apply tokeniser before the text, the emoji becomes 'u' and 's' which lost the original semantics
    tweet = demojize(tweet, delimiters=(" ", " "))

    if if_truncate:
        tweet = truncate_too_long_emojis(tweet)

    return tweet


def truncate_too_long_emojis(demojized_tweet, max_len=40):
    return " ".join(demojized_tweet.split()[:max_len])


def normalizeToken(token, normalise_mention=True, remove_url=False):
    """
    modified from https://github.com/VinAIResearch/BERTweet/blob/master/TweetNormalizer.py

    :param token:
    :return:
    """
    lowercased_token = token.lower()
    if normalise_mention is True and token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL" if remove_url is False else ""
    else:
        if token == "â€™":
            return "'"
        elif token == "â€¦":
            return "..."
        else:
            return token


def normalizeTweet(tweet, max_token_size: int = 60, lowercase=True, remove_emoji=False,
                   normalise_mention=True, remove_url=True, remove_punc=False, remove_stopwords=False,
                   return_str=False):
    """
    adapted from https://github.com/VinAIResearch/BERTweet/blob/master/TweetNormalizer.py

    a "soft" normalization strategy + simple lexical normalisation (deaccent) :

    by translating word tokens of user mentions and web/url links into special tokens @USER and HTTPURL, respectively,
    and converting emotion icon tokens into corresponding strings


    :param tweet:
    :param max_token_size: default to 60
        BERTTweet allow each tweet consists of at least 10 and at most 64 word tokens [Nguyen 2020]
            But empirically (in our experiment), the maximum token size is 59, otherwise it will raise index out of
            range error. Long token size is caused by few emoji tweets and our emoji normalisation method.
        DistilBERT accepts a max_sequence_length of 512 tokens
    :param lowercase:
    :param remove_emoji: True to remove emoji, false to convert emoji to text
    :param normalise_mention:
    :param remove_url: remove url since lots of tweets only contain url while we only want to match text
    :param remove_punc:remove punctuation to handle some variations
    :param return_str:
    :return:
    """
    if lowercase:
        tweet = tweet.lower()

    # deaccend and remove non-ascii unicode characters
    tweet = deaccent(tweet)
    if remove_emoji:
        tweet = give_emoji_free_text(tweet)
    else:
        tweet = emoji2text(tweet)

    tokens = nltk_tweet_tokenizer.tokenize(tweet.replace("â€™", "'").replace("â€¦", "..."))

    normTweet = " ".join(
        [normalizeToken(token, normalise_mention=normalise_mention, remove_url=remove_url) for token in tokens])
    # following encode/decode line of code can remove arabic content
    # normTweet = normTweet.encode('ascii', 'ignore').decode('utf-8', errors='surrogateescape')

    normTweet = normTweet.replace("cannot ", "can not ").replace("n't ", " n't ").replace("n 't ", " n't ").replace(
        "ca n't", "can't").replace("ai n't", "ain't")
    normTweet = normTweet.replace("'m ", " 'm ").replace("'re ", " 're ").replace("'s ", " 's ").replace("'ll ",
                                                                                                         " 'll ").replace(
        "'d ", " 'd ").replace("'ve ", " 've ")
    normTweet = normTweet.replace(" p . m .", "  p.m.").replace(" p . m ", " p.m ").replace(" a . m .",
                                                                                            " a.m.").replace(" a . m ",
                                                                                                             " a.m ")
    normTweet = re.sub(r",([0-9]{2,4}) , ([0-9]{2,4})", r",\1,\2", normTweet)
    normTweet = re.sub(r"([0-9]{1,3}) / ([0-9]{2,4})", r"\1/\2", normTweet)
    normTweet = re.sub(r"([0-9]{1,3})- ([0-9]{2,4})", r"\1-\2", normTweet)

    normed_tokens = normTweet.split()

    # remove duplicates
    normed_tokens = [i for i, _ in itertools.groupby(normed_tokens)]

    normed_tokens = normed_tokens[:max_token_size]
    if remove_punc:
        normed_tokens = list(punctuation_filter(normed_tokens))

    if remove_stopwords:
        normed_tokens = list(stop_words_filter(normed_tokens))

    if return_str:
        return " ".join(normed_tokens).strip()
    else:
        return normed_tokens


def is_minimum_tweet_filtering(tw_text, minimum_length = 5) -> Tuple:
    """simplified func to validate if the tweet text (info) length is below minimum threshold"""
    filtering = False

    tokens = minimum_text_norm(tw_text)

    if len(tokens) <= minimum_length:
        filtering = True
    return filtering, tokens


def minimum_text_norm(tw_text) -> List:
    """minimum text norm to remove emoji, handlers and 'RT' """
    # emoji_free_prop_tw = give_emoji_free_text(tw_text)
    # nltk_tweet_tokenizer.strip_handles = True
    # nltk_tweet_tokenizer.reduce_len = False
    # tokens = nltk_tweet_tokenizer.tokenize(emoji_free_prop_tw)
    tokens = normalizeTweet(tw_text, max_token_size=80, lowercase=False, remove_emoji=True, normalise_mention=True,
                   remove_url=True, remove_punc=True, remove_stopwords=False, return_str=False)

    # remove 'RT' if exists
    is_rt_tweet = 'RT' in tokens and tokens.index('RT') == 0
    if is_rt_tweet:
        tokens.pop(0)
    return tokens


def remove_rt_char(tw_text):
    txt_tokens = tw_text.split()
    is_rt_tweet = 'RT' in txt_tokens and txt_tokens.index('RT') == 0
    if is_rt_tweet:
        txt_tokens.pop(0)
    return " ".join(txt_tokens)
 
def generate_preprocess_dataset(dataset_path, output_path):
    rows = []
    lengths = []
    with open(dataset_path, 'r', encoding="utf8") as file:
    
        csvreader = csv.reader(file) 
        header = next(csvreader)
        print(header, len(header))
        try:
            for row in tqdm(csvreader):
                rows.append(row)
                lengths.append(len(row))
        except:
            pass
    new_df = pd.DataFrame(rows, columns=header)
    new_df.iloc[:100].to_csv("dummy.csv", index=False)
    new_df["content_process"] = new_df.content.apply(lambda x: normalizeTweet(preprocessing(x), max_token_size=80, lowercase=True, remove_emoji=True, normalise_mention=True,
                   remove_url=True, remove_punc=True, remove_stopwords=False, return_str=True))
    #new_df = new_df["content_process"]
    new_df["content_process"].to_csv( os.path.join(output_path,    os.path.basename(dataset_path.replace(".csv", "_processed.csv"))))


    cols = ["ids", "keywords", "score"]
    extractions = []

    for i in tqdm(range(len(new_df))):
        r.extract_keywords_from_text(new_df.content_process.values.tolist()[i])
        for score, w in r.get_ranked_phrases_with_scores():
            extractions.append([i, w, score])

    extraction_df = pd.DataFrame(extractions, columns=cols)
    extraction_df.to_csv(os.path.join(output_path,    os.path.basename(dataset_path.replace(".csv", "_extractions.csv"))), index=False)
    return new_df, extraction_df