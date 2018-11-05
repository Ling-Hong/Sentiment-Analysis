"""
Created on 2018-10-27 19:12:18
@author: Lynn.H
@description:
a naive bayes classifier for sentiment analysis (binary)

To improve the performance
- remove stopwords
- add geometric features
- hard-code rules

https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/
"""

import math
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from collections import Counter
import sys


# remove stop words from a string
# return a list
def clean_text_remove_sw(s):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(s.lower())
    return [i for i in words if not i in stop_words]

# keep stop words
def clean_text(s):
    words = word_tokenize(s.lower())
    return words

# x: a list of features
# y: a list of labels
def get_x_y(filename):
    input = []
    with open(filename) as f:
        input = [clean_text(i) for i in f.readlines()]

    feature = [i[1:] for i in input]
    label = [i[0] for i in input]
    return feature, label

def reduce_dimension(lst):
    return [j for i in lst for j in i]

# return 1D list of words
def categorize_text(filename, label1, label2):
    x, y = get_x_y(filename)
    x1 = reduce_dimension([x[i] for i in range(len(x)) if y[i]==label1])
    x2 = reduce_dimension([x[i] for i in range(len(x)) if y[i]==label2])
    return x1, x2, y

def count_words(l1,l2):
    count1 = Counter(l1)
    only_l2 = set(l2) - set(l1)
    only_l2_count = {k:v-1 for k,v in Counter(only_l2).items()}
    count1.update(only_l2_count)

    count2 = Counter(l2)
    only_l1 = set(l1) - set(l2)
    only_l1_count = {k:v-1 for k,v in Counter(only_l1).items()}
    count2.update(only_l1_count)
    return count1, count2


def add_one_smoothing(count1, count2, V):
    count1 = {k:math.log((v+1)/(sum(count1.values())+len(V)),2) for k,v in count1.items()}
    count2 = {k:math.log((v+1)/(sum(count2.values())+len(V)),2) for k,v in count2.items()}
    return count1, count2

def train_nb(filename,label_list):
    red, blue, labels = categorize_text(filename, label_list[0], label_list[1])
    all_words = set(red) | set(blue)
    V = all_words

    # calculate P(c): prior probability for each class
    red_prior = math.log(sum([i == 'red' for i in labels])/ len(labels),2)
    blue_prior = math.log(sum([i == 'blue' for i in labels])/ len(labels),2)

    # calculate P(w|c)
    red_count, blue_count = count_words(red, blue)
    new_red_count, new_blue_count = add_one_smoothing(red_count, blue_count, V)
    return red_prior, blue_prior, new_red_count, new_blue_count, V

# C -- a list of all labels
# V -- vocabulary of documents
def test_nb(testdoc, logprior, loglikelihood, V):

    # train_input is a list of sentences, each sentence is a list of words
    train_input, train_label = get_x_y(testdoc)

    # calcualte the probability for "red" class
    red_loglikelihood = loglikelihood[0]

    # calcualte the probability for "blue" class
    blue_loglikelihood = loglikelihood[1]

    result = []
    result_detailed = []
    for s in train_input:
        red_prob = logprior[0]
        blue_prob = logprior[1]
        for w in s:
            # ignore unkown words
            if w in V:
                red_prob += red_loglikelihood[w]
                blue_prob += blue_loglikelihood[w]
        if red_prob>blue_prob:
            tag = 'red'
        elif red_prob<blue_prob:
            tag = 'blue'
        else:
            tag = 'both'
        result.append(tag)
        result_detailed.append([tag, red_prob, blue_prob])
    return train_label, result, result_detailed


def evaluate(y, y_predict):
    y = [1 if i=='red' else 0  for i in y]
    y_predict = [1 if i == 'red' else 0 for i in y_predict]

    tp = sum([1 for i,j in zip(y, y_predict) if i==1 and j==1])
    fp = sum([1 for i,j in zip(y, y_predict) if i==0 and j==1])
    fn = sum([1 for i,j in zip(y, y_predict) if i==1 and j==0])
    tn = sum([1 for i, j in zip(y, y_predict) if i == 0 and j == 0])

    # accuracy
    acc = (tp + tn)/len(y)

    # precision
    red_precision = tp / (tp+fp)
    blue_precision = tn / (tn+fn)

    # recall
    red_recall = tp / (tp+fn)
    blue_recall = tn / (tn+fp)

    return acc, red_precision, red_recall, blue_precision, blue_recall


def main():
    train1 = sys.argv[1]
    test1 = sys.argv[2]
    train2 = sys.argv[3]
    test2 = sys.argv[4]

    # train the model
    red_prior1, blue_prior1, new_red_count1, new_blue_count1, V1 = train_nb(train1,['red','blue'])

    # test result
    test_result1, test_label1, test_result_detailed1 = test_nb(test1, [red_prior1,blue_prior1], [new_red_count1,new_blue_count1], V1)
    # evaluate the result
    acc1, red_precision1, red_recall1, blue_precision1, blue_recall1 = evaluate(test_result1, test_label1)

    # train the model
    red_prior2, blue_prior2, new_red_count2, new_blue_count2, V2 = train_nb(train2,['red','blue'])
    # test result
    test_result2, test_label2, test_result_detailed2 = test_nb(test2, [red_prior2,blue_prior2], [new_red_count2,new_blue_count2], V2)
    # evaluate the result
    acc2, red_precision2, red_recall2, blue_precision2, blue_recall2 = evaluate(test_result2, test_label2)

    with open("task1.txt","w") as f:
        f.write("overall accuracy\n")
        f.write(str(round(acc1,1)))
        f.write("\n")
        f.write("precision for red\n")
        f.write(str(round(red_precision1,1)))
        f.write("\n")
        f.write("recall for red\n")
        f.write(str(round(red_recall1,1)))
        f.write("\n")
        f.write("precision for blue\n")
        f.write(str(round(blue_precision1,1)))
        f.write("\n")
        f.write("recall for blue\n")
        f.write(str(round(blue_recall1,1)))
        f.write("\n")
        f.write("\n")
        f.write("overall accuracy\n")
        f.write(str(round(acc2,1)))
        f.write("\n")
        f.write("precision for red\n")
        f.write(str(round(red_precision2,1)))
        f.write("\n")
        f.write("recall for red\n")
        f.write(str(round(red_recall2,1)))
        f.write("\n")
        f.write("precision for blue\n")
        f.write(str(round(blue_precision2,1)))
        f.write("\n")
        f.write("recall for blue\n")
        f.write(str(round(blue_recall2,1)))

    # with open('result_detail1.txt','w') as f:
    #     for s in test_result_detailed1:
    #         f.write("label: {} red_prob: {} blue_prob: {} diff: {}\n".format(s[0], s[1], s[2], abs(s[2]-s[1])))
    #
    # with open('result_detail2.txt', 'w') as f:
    #     for s in test_result_detailed2:
    #         f.write("label: {} red_prob: {} blue_prob: {} diff: {}\n".format(s[0], s[1], s[2], abs(s[2]-s[1])))


if __name__ == "__main__":
    main()

