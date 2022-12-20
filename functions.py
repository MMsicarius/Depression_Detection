import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import nltk
import math
import matplotlib
from collections import Counter
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from wordcloud import WordCloud
from nltk.corpus import stopwords, words
from nltk.sentiment import SentimentIntensityAnalyzer


def progress_bar(progress, data_length):
    result = progress / data_length * 100
    result = math.floor(result)
    if result != 100:
        return print(str(result) + "%", end="\r")
    else:
        return print("completed!")

def most_common(list):
    print(list)
    most_freq = Counter(list).most_common(100)
    return most_freq

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


def correction(input_text, mode):
    if mode == 1:
        unfiltered_text = TextBlob(input_text)
        unfiltered_text.correct()
        return str(unfiltered_text)
    else:
        return input_text


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

    def load_corpus_filtered(self):
        stop_words = set(stopwords.words("english"))
        english_words = set(words.words())
        data = [x for x in self.dataset["text"]]
        token_result = []
        progress = 0
        data_length = len(data)
        for x in data:
            x = correction(x, 1)
            filter_result = []
            result = nltk.word_tokenize(x)
            for y in result:
                if y in english_words and y not in stop_words:
                    filter_result.append(y)
            token_result.append(filter_result)
            progress += 1
            progress_bar(progress, data_length)

        self.dataset["token"] = token_result
        print("dataset tokenised!")

    def sentiment_analysis_vader(self):
        depressed_sample = self.dataset.query("depressed == 1")
        not_depressed_sample = self.dataset.query("depressed == 0")
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
        print("sentiment analysis processed!")

    def sentiment_analysis_vader_nosplit(self):
        results_neg = []
        results_neu = []
        results_pos = []
        results_com = []
        progress = 0
        data_length = len(self.dataset)
        depressed_sample = self.dataset.query("depressed == 1")
        not_depressed_sample = self.dataset.query("depressed == 0")

        sentiment_analysis = SentimentIntensityAnalyzer()
        print("Processing dataset with no sentence splitting")
        for x in self.dataset.index:
            sample = self.dataset.iloc[[x], [0]]
            result = sentiment_analysis.polarity_scores(str(sample))
            results_neg.append(result["neg"])
            results_neu.append(result["neu"])
            results_pos.append(result["pos"])
            results_com.append(result["compound"])
            progress += 1
            progress_bar(progress, data_length)

        self.dataset["vader_negative_nosplit"] = results_neg
        self.dataset["vader_neutral_nosplit"] = results_neu
        self.dataset["vader_positive_nosplit"] = results_pos
        self.dataset["vader_compound_nosplit"] = results_com
        print("process completed")

    def sentiment_word_analysis(self):
        depressed_breakdown = self.dataset[self.dataset["depressed"] == 1]
        not_depressed_breakdown = self.dataset[self.dataset["depressed"] == 0]
        combined_depressed_responses = []
        combined_not_depressed_responses = []
        english_words = set(words.words())
        stop_words = set(stopwords.words("english"))

        result = depressed_breakdown.token.tolist()
        progress = 0
        data_length = len(result)
        for x in result:
            combined_depressed_responses = combined_depressed_responses + x
            progress += 1
            progress_bar(progress, data_length)

        result = not_depressed_breakdown.token.tolist()
        progress = 0
        data_length = len(result)
        for x in result:
            combined_not_depressed_responses = combined_not_depressed_responses + x
            progress += 1
            progress_bar(progress, data_length)

        most_common_depressed_words = Counter(" ".join(combined_depressed_responses).split()).most_common(100)#most_common(combined_depressed_responses)
        most_common_not_depressed_words = Counter(" ".join(combined_not_depressed_responses).split()).most_common(100)#most_common(combined_not_depressed_responses)



        most_common_depressed_words_individuals = []
        most_common_depressed_words_numbers = []
        most_common_not_depressed_words_individuals = []
        most_common_not_depressed_words_numbers = []

        for i in most_common_depressed_words:
            most_common_depressed_words_individuals.append(i[0])
            most_common_depressed_words_numbers.append(i[1])

        for i in most_common_not_depressed_words:
            most_common_not_depressed_words_individuals.append(i[0])
            most_common_not_depressed_words_numbers.append(i[1])

        depressed_words_unique = []
        for i in most_common_depressed_words_individuals:
            if i not in most_common_not_depressed_words_individuals:
                depressed_words_unique.append(i)

        not_depressed_words_unique = []
        for i in most_common_not_depressed_words_individuals:
            if i not in most_common_depressed_words_individuals:
                not_depressed_words_unique.append(i)


        print(most_common_depressed_words_individuals)
        print(most_common_not_depressed_words_individuals)



    def sentiment_analysis(self):
        depressed_breakdown = self.dataset[self.dataset["depressed"] == 1]
        not_depressed_breakdown = self.dataset[self.dataset["depressed"] == 0]


        print("Description of depressed dataset")
        print(depressed_breakdown.describe())
        print("Description of the not depressed dataset")
        print(not_depressed_breakdown.describe())

        ax = WordCloud(background_color="white", width=1500, height=1500).generate(
            str(depressed_breakdown["token"]))
        plt.axis("off")
        plt.imshow(ax)
        plt.show()
        ax = WordCloud(background_color="white", width=1500, height=1500).generate(
            str(not_depressed_breakdown["token"]))
        plt.axis("off")
        plt.imshow(ax)
        plt.show()
        ax = self.dataset.plot.hist(column=["vader_negative"], by="depressed")
        plt.show()
        ab = self.dataset.plot.hist(column=["vader_neutral"], by="depressed")
        plt.show()
        ac = self.dataset.plot.hist(column=["vader_positive"], by="depressed")
        plt.show()
        ad = self.dataset.plot.hist(column=["vader_negative_nosplit"], by="depressed")
        plt.show()
        ae = self.dataset.plot.hist(column=["vader_neutral_nosplit"], by="depressed")
        plt.show()
        af = self.dataset.plot.hist(column=["vader_positive_nosplit"], by="depressed")
        plt.show()
        ae = self.dataset.boxplot(column=["vader_negative_nosplit", "vader_neutral_nosplit",
                                               "vader_positive_nosplit", "vader_compound_nosplit"], by="depressed")
        plt.show()

        #  TODO concordance
        #  TODO Collocation

    def assess_vader_diff(self):
        self.dataset["vader_difference_neg"] = self.dataset["vader_negative"] - self.dataset["vader_negative_nosplit"]
        self.dataset["vader_difference_neu"] = self.dataset["vader_neutral"] - self.dataset["vader_neutral_nosplit"]
        self.dataset["vader_difference_pos"] = self.dataset["vader_positive"] - self.dataset["vader_positive_nosplit"]
        self.dataset["vader_difference_com"] = self.dataset["vader_compound"] - self.dataset["vader_compound_nosplit"]

    def print_dataset(self):
        print(self.dataset.head())
        print(self.dataset.describe())

    def loop_test(self):
        result = self.dataset.query("depressed == 1").sample(frac=.1)
        for x in result.index:
            print(result["token"][x])

    def model_tests(self):
        detokenised = []
        for x in self.dataset.token:
            detokenised.append(" ".join(x))
        self.dataset["text_train"] = detokenised
        X = self.dataset.text_train
        y = self.dataset.depressed

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        vector = CountVectorizer(max_features=1000, binary=True)

        X_train_vectorised = vector.fit_transform(X_train)

        nb = MultinomialNB()

        nb.fit(X_train_vectorised, y_train)

        X_test_vetorised = vector.transform(X_test)

        y_predict = nb.predict(X_test_vetorised)



        print("Naive Brayers")
        print("Accuracy:" + " " + str((round((accuracy_score(y_test, y_predict) * 100), 2))) + "%")
        print("F1 score" + " " + str((round((f1_score(y_test, y_predict) * 100), 2))))

        model = LogisticRegression()
        model.fit(X_train_vectorised, y_train)
        y_predict = model.score(X_train_vectorised, y_train)
        print("Logistic Regression")
        print("R2:" + " " + str(y_predict))

