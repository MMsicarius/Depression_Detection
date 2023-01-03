from nltk.corpus import words

import functions
import nltk

data = functions.Functions()

data.load_dataset()

data.load_corpus_filtered()

data.sentiment_analysis_vader()

data.sentiment_analysis_vader_nosplit()

data.assess_vader_diff()

data.model_tests()

data.print_dataset()

data.average_response_size()

data.sentiment_word_analysis()

data.sentiment_analysis()


