import pandas as pd
import numpy as np
import nltk
import math
from nltk.sentiment import SentimentIntensityAnalyzer

def progress_bar(progress, data_length):
    result = progress / data_length * 100
    result = math.floor(result)
    if result != 100:
        return print(str(result) + "%", end="\r")
    else:
        return print("completed!")

def sample_size(data, sample_size_number):
    result = data
    return result.sample(frac=sample_size_number)

def sample_loop(data):
    sentiment_analysis = SentimentIntensityAnalyzer()
    method_result = []
    progress = 0
    data_length = len(data)
    for x in data.index:
        search_result = data["token"][x]
        temp_negative = []
        temp_neutral = []
        temp_positive = []
        temp_compound = []
        answer = []
        response_length = len(search_result)
        sentence = 0
        while sentence + 20 < response_length:
            if (sentence - 1) + 20 <= response_length:
                response = " ".join(search_result[sentence:(sentence + 20)])
            else:
                response = search_result[sentence:response_length]
            result = sentiment_analysis.polarity_scores(response)
            temp_negative.append(result.get("neg"))
            temp_neutral.append(result.get("neu"))
            temp_positive.append(result.get("pos"))
            temp_compound.append(result.get("compound"))
            sentence += 20


        answer = 0
        for i in temp_negative:
            answer = answer + i
        if answer != 0:
            final_answer_neg = answer / len(temp_negative)
        else:
            final_answer_neg = 0
        method_result.append(final_answer_neg)
        answer = 0
        for i in temp_neutral:
            answer = answer + i
        if answer != 0:
            final_answer_neu = answer / len(temp_neutral)
        else:
            final_answer_neu = 0
        method_result.append(final_answer_neu)
        answer = 0
        for i in temp_positive:
            answer = answer + i
        if answer != 0:
            final_answer_pos = answer / len(temp_positive)
        else:
            final_answer_pos = 0
        method_result.append(final_answer_pos)
        answer = 0
        for i in temp_compound:
            answer = answer + i
        if answer != 0:
            final_answer_compound = answer / len(temp_compound)
        else:
            final_answer_compound = 0
        method_result.append(final_answer_compound)
        progress += 1
        progress_bar(progress, data_length)
    return method_result

def loop_results(data, position):
    result = []
    position = 0
    iterator = 0
    length = len(data) / 4
    for x in data:
        if iterator % 4 == position:
            result.append(x)
        else:
            pass
        iterator += 1
    return result

class Functions:
    def __init__(self, dataset="depression_dataset_reddit.csv"):
        self.dataset = dataset


    def load_dataset(self):
        self.dataset = pd.read_csv("depression_dataset_reddit.csv")
        self.dataset.rename(columns={"clean_text": "text", "is_depression": "depressed"}, inplace=True)
        print("Loaded dataset!")

    def load_corpus(self):
        data = [x for x in self.dataset["text"]]
        token_result = []
        for x in data:
            token_result.append(nltk.word_tokenize(x))

        self.dataset["token"] = token_result

    def sentiment_analysis(self):
        depressed_sample = self.dataset.query("depressed == 1").sample(frac=1, replace=True, random_state=1)
        not_depressed_sample = self.dataset.query("depressed == 0").sample(frac=1, replace=True, random_state=1)
        iterator = 0
        vader_negative = []
        vader_neutral = []
        vader_positive = []
        vader_compound = []
        print("processing depressed answers")
        depressed_answers = sample_loop(depressed_sample)
        vader_negative = loop_results(depressed_answers, 0)
        vader_neutral = loop_results(depressed_answers, 1)
        vader_positive = loop_results(depressed_answers, 2)
        vader_compound = loop_results(depressed_answers, 3)
        print("processing not depressed answers")
        not_depressed_answers = sample_loop(not_depressed_sample)
        vader_negative.extend(loop_results(not_depressed_answers, 0))
        vader_neutral.extend(loop_results(not_depressed_answers, 1))
        vader_positive.extend(loop_results(not_depressed_answers, 2))
        vader_compound.extend(loop_results(not_depressed_answers, 3))



        self.dataset["vader_negative"] = vader_negative
        self.dataset["vader_neutral"] = vader_neutral
        self.dataset["vader_positive"] = vader_positive
        self.dataset["vader_compound"] = vader_compound
        print("sentiment analysis complete")


    def print_dataset(self):
        print(self.dataset.head())
        print(self.dataset.describe())

    def loop_test(self):
        result = self.dataset.query("depressed == 1").sample(frac=.1)
        for x in result.index:
            print(result["token"][x])

