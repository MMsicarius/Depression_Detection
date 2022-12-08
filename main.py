import nltk
import pandas as pd
import numpy as np
import functions

data = functions.Functions()

data.load_dataset()

data.load_corpus()

data.sentiment_analysis_vader()

data.sentiment_analysis()

data2 = functions.Tweets()

data2.load_dataset()

data2.print_dataset()








