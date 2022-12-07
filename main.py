import nltk
import pandas as pd
import numpy as np
import functions

data = functions.Functions()

data.load_dataset()

data.load_corpus()

data.sentiment_analysis()

data.print_dataset()







