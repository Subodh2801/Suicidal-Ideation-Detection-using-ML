"""
Feature-mask helpers built around sklearn-genetic GeneticSelectionCV.

Reads/writes boolean masks under model/ when invoked from the main app or
notebook. Sai Subodh — see README for usage.
"""
from sklearn.ensemble import RandomForestClassifier
from genetic_selection import GeneticSelectionCV
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer #loading tfidf vector
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

global original_X, Y, linguistic_X

def loadData():
    global original_X, Y, linguistic_X
    original_X = np.load("model/X.npy")
    Y = np.load("model/Y.npy")
    statistics = np.load("model/statistics.npy")
    linguistic_X = np.load("model/linguistic.npy")

    original_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=stop_words, use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=196)
    original_X = original_vectorizer.fit_transform(original_X).toarray()
    original_X = np.hstack([original_X, statistics])
    linguistic_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=stop_words, use_idf=True, smooth_idf=False, norm=None, decode_error='replace', max_features=160)
    linguistic_X = linguistic_vectorizer.fit_transform(linguistic_X).toarray()
    

def runOriginalGA():
    global original_X, Y
    if os.path.exists("model/original_ga.npy"):
        selector = np.load("model/original_ga.npy")
    else:
        estimator = RandomForestClassifier()
        #defining genetic alorithm object
        selector = GeneticSelectionCV(estimator, cv=5, verbose=1, scoring="accuracy", max_features=86, n_population=10, crossover_proba=0.5, mutation_proba=0.2,
                                      n_generations=5, crossover_independent_proba=0.5, mutation_independent_proba=0.05, tournament_size=3, n_gen_no_change=10,
                                      caching=True, n_jobs=-1)
        ga_selector = selector.fit(original_X, Y) #train with GA weights
        selector = ga_selector.support_
        np.save("model/original_ga", selector)
    return selector

def runLinguisticGA():
    global linguistic_X, Y
    if os.path.exists("model/linguistic_ga.npy"):
        selector = np.load("model/linguistic_ga.npy")
    else:
        estimator = RandomForestClassifier()
        #defining genetic alorithm object
        selector = GeneticSelectionCV(estimator, cv=5, verbose=1, scoring="accuracy", max_features=59, n_population=10, crossover_proba=0.5, mutation_proba=0.2,
                                      n_generations=5, crossover_independent_proba=0.5, mutation_independent_proba=0.05, tournament_size=3, n_gen_no_change=10,
                                      caching=True, n_jobs=-1)
        ga_selector = selector.fit(linguistic_X, Y) #train with GA weights
        selector = ga_selector.support_
        np.save("model/linguistic_ga", selector)
    return selector

if __name__ == "__main__":
    loadData()
    original_ga = runOriginalGA()
    linguistic_ga = runLinguisticGA()
    original_X = original_X[:,original_ga]
    linguistic_X = linguistic_X[:,linguistic_ga]
    print(original_X)
    print(linguistic_X)
    print(original_X.shape)
    print(linguistic_X.shape)
