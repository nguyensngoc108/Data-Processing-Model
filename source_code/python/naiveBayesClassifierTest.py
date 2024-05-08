from email import message
import pandas as pd
import numpy as np
import sys


class NaiveBayesFilter:
    def __init__(self):
        self.data = []
        self.vocabulary = []  # returns tuple of unique words
        self.proba = []
        self.prob = []
        self.p_spam = 0  # Probability of Spam
        self.p_ham = 0  # Probability of Ham
        # Initiate parameters
        self.parameters_spam = {
            unique_word: 0 for unique_word in self.vocabulary}
        self.parameters_ham = {
            unique_word: 0 for unique_word in self.vocabulary}

    def fit(self, X, y):
        for sms in X:
            for word in sms:
                self.vocabulary.append(word)
        self.vocabulary = list(set(self.vocabulary))

        word_count_per_sms = {unique_word: [0] * len(X)
                              for unique_word in self.vocabulary}
        for index, sms in enumerate(X):
            for word in sms:
                word_count_per_sms[word][index] += 1
        word_count = pd.DataFrame(word_count_per_sms)

        self.data = pd.concat([y, X, word_count], axis=1)
        return self.data

    def predict_proba(self, X, y):
        spam_messages = self.data[self.data['Label'] == 'spam']
        ham_messages = self.data[self.data['Label'] == 'ham']

        self.p_spam = len(spam_messages) / len(self.data)
        self.p_ham = len(ham_messages) / len(self.data)

        no_of_words_per_spam = spam_messages['SMS'].apply(len)
        no_of_spam = no_of_words_per_spam.sum()

        no_of_words_per_ham = ham_messages['SMS'].apply(len)
        no_of_ham = no_of_words_per_ham.sum()

        no_of_vocabulary = len(self.vocabulary)

        alpha = 1

        for word in self.vocabulary:
            no_of_word_given_spam = spam_messages[word].sum()
            p_word_given_spam = (no_of_word_given_spam + alpha) / \
                (no_of_spam + alpha * no_of_vocabulary)
            self.parameters_spam[word] = p_word_given_spam

            no_of_word_given_ham = ham_messages[word].sum()
            p_word_given_ham = (no_of_word_given_ham + alpha) / \
                (no_of_ham + alpha * no_of_vocabulary)
            self.parameters_ham[word] = p_word_given_ham

        p_spam_given_message = self.p_spam
        p_ham_given_message = self.p_ham

        message_proba = {'Spam Proba': [0] * len(X), 'Ham Proba': [0] * len(X)}

        for index, sms in enumerate(X):
            for word in sms:
                if word in self.parameters_spam:
                    p_spam_given_message *= self.parameters_spam[word]
                if word in self.parameters_ham:
                    p_ham_given_message *= self.parameters_ham[word]
            message_proba['Spam Proba'][index] = p_spam_given_message
            message_proba['Ham Proba'][index] = p_ham_given_message
        predicted_proba = pd.DataFrame(message_proba)

        self.proba = pd.concat([y, X, predicted_proba], axis=1)
        return self.proba
