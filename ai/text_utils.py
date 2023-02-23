# -*- coding: utf-8 -*-
import os, sys

import re
import nltk
import numpy as np
import json
from tqdm import tqdm
from gensim.utils import deaccent
from nltk.tokenize import TweetTokenizer
from emoji import demojize, get_emoji_regexp
import itertools
import string
from nltk.corpus import stopwords
from typing import List
from gensim.models import fasttext
from transformers.tokenization_utils_base import BatchEncoding
from transformers.models.bert.modeling_tf_bert import TFBertModel
import tensorflow as tf


fasttext_model_path = os.environ.get("fasttext_path",
                                     "gs://rumor_detection_dev/embedding_models/fasttext/crawl-300d-2M-subword.bin")

try:
    _stop_words = stopwords.words('english')
except LookupError:
    nltk.download('stopwords')
    _stop_words = stopwords.words('english')

nltk_tweet_tokenizer = TweetTokenizer(strip_handles=False, reduce_len=True)
punctuation_filter = lambda t: filter(lambda a: a not in string.punctuation, t)
stop_words_filter = lambda t: filter(lambda a: a not in _stop_words, t)


def give_emoji_free_text(text):
    if isinstance(text, bytes):
        text = text.decode('utf8')
    return get_emoji_regexp().sub(r'', text)


def truncate_too_long_emojis(demojized_tweet, max_len=40):
    return " ".join(demojized_tweet.split()[:max_len])


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
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token


def simple_text_norm(tweet: str, remove_url=True):
    tokens = tweet.strip().lower().split()
    if len(tokens) == 0:
        return ''

    normTweet = " ".join(
        [normalizeToken(token, normalise_mention=False, remove_url=remove_url) for token in tokens])

    return normTweet


def normalizeTweet(tweet, max_token_size: int = 60, lowercase=True, remove_emoji=False,
                   normalise_mention=True, remove_url=True, remove_punc=False, remove_stopwords=False,
                   return_str=False):
    """
    adapted from https://github.com/VinAIResearch/BERTweet/blob/master/TweetNormalizer.py

    a "soft" normalization strategy + simple lexical normalisation (deaccent) :

    by translating word tokens of user mentions and web/url links into special tokens @USER and HTTPURL, respectively,
    and converting emotion icon tokens into corresponding strings


    :param tweet:
    :param max_token_size:
    :param lowercase:
    :param remove_emoji:
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

    tokens = nltk_tweet_tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))

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


def load_fasttext_model(fasttext_model_path):
    return fasttext.load_facebook_vectors(fasttext_model_path)


def build_vocab(word_list1: List[str], word_list2: List[str]) -> List:
    vocab_idx = set(word_list1 + word_list2)
    return list(vocab_idx)


def word_overlapping_ratio(claim_text, doc_text):
    """check overlap between claim_text (claim) and doc_text (document).
    1. For the word overlapping ratio, we calculate the number of overlapping tokens between T and H and
    normalize it by dividing it by the number of tokens in H [Bos 2005, Wang 2009]

    Wang, R., & Zhang, Y. (2009, August). Recognizing textual relatedness with predicate-argument structures.
    In Proceedings of the 2009 Conference on Empirical Methods in Natural Language Processing (pp. 784-792).

    Bos, J., & Markert, K. (2005, October). Recognising textual entailment with logical inference.
    In Proceedings of Human Language Technology Conference and Conference on Empirical Methods in Natural Language Processing (pp. 628-635).

    2. word overlap = overlap(H,T) / len(H) + len(T)
    Agichtein, E., Askew, W., & Liu, Y. (2008, November). Combining Lexical, Syntactic, and Semantic Evidence for
    Textual Entailment Classification. In TAC. https://tac.nist.gov/publications/2008/participant.papers/Emory.proceedings.pdf

    Note: formula 1 make sense here allow us to get insight of the word overlap in claim/hypothesis H by document text (T)
    """
    norm_text1_tokens = normalizeTweet(claim_text, normalise_mention=False, remove_punc=True, remove_stopwords=True)
    norm_text2_tokens = normalizeTweet(doc_text, normalise_mention=False, remove_punc=True, remove_stopwords=True)

    overlapping_ratio = 0
    if len(norm_text1_tokens) == 0 or len(norm_text2_tokens) == 0:
        return overlapping_ratio

    word_vocabs = build_vocab(norm_text1_tokens, norm_text2_tokens)

    norm_text1_tokens_idx = [word_vocabs.index(token) for token in norm_text1_tokens]
    norm_text2_tokens_idx = [word_vocabs.index(token) for token in norm_text2_tokens]
    overlaps = [token_idx for token_idx in norm_text1_tokens_idx if token_idx in norm_text2_tokens_idx]

    # overlapping_ratio = round(len(overlaps) / len(norm_text2_tokens_idx), 5)
    overlapping_ratio = round(len(overlaps) / len(norm_text1_tokens_idx), 3)

    return overlapping_ratio


def text_transformer_embedding(texts_tokenised_encoded_input: BatchEncoding, transformer_embedding_model: TFBertModel) -> np.ndarray:
    tokens_input_ids = texts_tokenised_encoded_input["input_ids"]
    tokens_attention_masks = texts_tokenised_encoded_input["attention_mask"]

    return text_transformer_embedding_from_ids_attnmask(tokens_input_ids, tokens_attention_masks,
                                                        transformer_embedding_model)


def text_transformer_embedding_from_ids_attnmask(tokens_input_ids, tokens_attention_masks,
                                                 transformer_embedding_model: TFBertModel):
    all_embeddings = []

    for input_ids, attention_mask in zip(tokens_input_ids, tokens_attention_masks):
        if not isinstance(input_ids, np.ndarray):
            input_ids = input_ids.numpy()
        input_ids = tf.convert_to_tensor([input_ids])
        if not isinstance(attention_mask, np.ndarray):
            attention_mask = attention_mask.numpy()
        attention_mask = tf.convert_to_tensor([attention_mask])
        text_embedding = transformer_embedding_model(**{'input_ids': input_ids, 'attention_mask': attention_mask})
        text_embedding = text_embedding["last_hidden_state"].numpy()
        text_embedding = text_embedding[0]
        # reduce GPU memory by moving GPU tensors hosted in GPU to CPU
        all_embeddings.append(text_embedding)
    # texts_embeddings = transformer_embedding_model(**text_tokenised_input)
    # texts_embeddings["last_hidden_state"].numpy()
    return np.asarray(all_embeddings)



def test_text_transformer_embedding():
    data_dir = "C:\\data\\fake_news_data\\Factify\\public_folder"
    image_dir = "C:\\data\\fake_news_data\\Factify\\val_images"
    dataset_name = "val"

    from data_load import load_factify_data_4_multimodal_snli
    from transformers import AutoTokenizer, TFAutoModel

    multimodal_df = load_factify_data_4_multimodal_snli(data_dir, file_name=dataset_name, image_dir=image_dir)
    val_text_left_list = multimodal_df["text_left"].values
    val_text_right_list = multimodal_df["text_right"].values

    val_text_right_list = val_text_right_list[:20]

    embedding_model_name = "bert-base-uncased"
    transformer_embedding_model = TFAutoModel.from_pretrained(embedding_model_name)
    transformer_tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)

    val_text_right_encoded_input = transformer_tokenizer(val_text_right_list.tolist(), padding='max_length',
                                                         truncation=True, max_length=512, return_tensors='tf')
    val_text_right_embeddings = text_transformer_embedding(val_text_right_encoded_input, transformer_embedding_model)
    print(val_text_right_embeddings.shape)

if __name__ == '__main__':
    claim_text = "For Democrats, Stacey Abrams sends key message on gender and race in SOTU response. https://t.co/ThlfBjq8mQ https://t.co/6HBKawYAmF"
    doc_text = "Democratic Party rising star Stacey Abrams sharply criticized the Trump administration and Republican leadership on Tuesday night in her response to the President’s State of the Union address. "

    test_text_transformer_embedding()

    # overlap_ratio = word_overlapping_ratio(claim_text, doc_text)
    # print(overlap_ratio)
    # from tensorflow.keras.preprocessing.text import text_to_word_sequence
    #
    # text_seq = text_to_word_sequence(claim_text)
    # print(text_seq)
    # print("=====")
    # print(simple_text_norm(claim_text))
    # fasttext_model = load_fasttext_model("C:\\models\\fasttext\\crawl-300d-2M-subword.bin")
    # print(fasttext_model['sotu'])
    # print(fasttext_model['https://t.co/ThlfBjq8mQ'])
