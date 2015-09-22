__author__ = 'Vardhaman'
#Implementation of multivariate bernoulli naive based classifier
import sys
import csv
import math
import copy
import time
import numpy as np
from operator import itemgetter, attrgetter
import matplotlib.pyplot as plt
from collections import Counter
from numpy.linalg import inv
prior_prob_dict = {}# prior prob value for each class
doc_class = {}#doc id and class value
word_dict = {}#dic with word and freq over the entire corpus
test_doc_class = {}#dic with test doc id and its corresponding class value
doc_dict = {}#construct test doc with docid as key and word n count as another dict
class_dict = {} # no of docs belonging to a class

#method to get a dictionary for each word and its frequency over the entire corpus
def vocab_fre(train_file):
    f = open(train_file,'r')
    for line in f.readlines():
        words = line.split()
        if word_dict.has_key(words[1]):
                word_dict[words[1]] = int(word_dict[words[1]]) + int(words[2])
        else:
            word_dict[words[1]] = int(words[2])
    print "dictionary",word_dict

#method to cal prior probability for each class
def cal_prior_prob_for_class(label_file):
    f = open(label_file, 'r')
    count = 1
    for line in f.readlines():
        doc_class[count] = line.strip("\n")
        if(class_dict.has_key(line.strip("\n"))):
            class_dict[line.strip("\n")] = class_dict.get(line.strip("\n"))+1
        else:
            class_dict[line.strip("\n")] = 1
        count +=1
    sum_val = sum(class_dict.values())
    #print class_dict
    for key in class_dict.keys():
        prior_prob_dict[key.strip()] = math.log(class_dict[key.strip()]) - math.log(sum_val)
        #prior_prob_dict[key.strip()] = float(class_dict[key.strip()])/sum_val
    #print prior_prob_dict
    #print doc_class

#word and associated count for that word for each class
def add_word_to_dict(class_dict,word,count):
    if class_dict.has_key(word):
        class_dict[word] = int(class_dict.get(word)) + int(count)
    else:
        class_dict[word] = int(count)

# this logic is horrible right now, need to improve it soon
# dicitonary with each class id and corresponding count for no of docs in which the word occurs
#this is done because we are doing bernoulli
def add_word_to_appropriate_class_dict(words):
    doc_class_val= int(doc_class[int(words[0])])
    #print(doc_class_val)
    if doc_class_val == 1:
        if class_1.has_key(words[1]):
            class_1[words[1]] = int(class_1.get(words[1])) + 1
        else:
            class_1[words[1]] =  1
    elif doc_class_val == 2:
        if class_2.has_key(words[1]):
            class_2[words[1]] = int(class_2.get(words[1])) + 1
        else:
            class_2[words[1]] = 1
    elif doc_class_val == 3:
        if class_3.has_key(words[1]):
            class_3[words[1]] = int(class_3.get(words[1])) + 1
        else:
            class_3[words[1]] = 1
    elif doc_class_val == 4:
        if class_4.has_key(words[1]):
            class_4[words[1]] = int(class_4.get(words[1])) + 1
        else:
            class_4[words[1]] = 1
    elif doc_class_val == 5:
        if class_5.has_key(words[1]):
            class_5[words[1]] = int(class_5.get(words[1])) + 1
        else:
            class_5[words[1]] = 1
    elif doc_class_val == 6:
        if class_6.has_key(words[1]):
            class_6[words[1]] = int(class_6.get(words[1])) + 1
        else:
            class_6[words[1]] = 1
    elif doc_class_val == 7:
        if class_7.has_key(words[1]):
            class_7[words[1]] = int(class_7.get(words[1])) + 1
        else:
            class_7[words[1]] = 1
    elif doc_class_val == 8:
        if class_8.has_key(words[1]):
            class_8[words[1]] = int(class_8.get(words[1])) + 1
        else:
            class_8[words[1]] = 1
    elif doc_class_val == 9:
        if class_9.has_key(words[1]):
            class_9[words[1]] = int(class_9.get(words[1])) + 1
        else:
            class_9[words[1]] = 1
    elif doc_class_val == 10:
        if class_10.has_key(words[1]):
            class_10[words[1]] = int(class_10.get(words[1])) + 1
        else:
            class_10[words[1]] = 1
    elif doc_class_val == 11:
        if class_11.has_key(words[1]):
            class_11[words[1]] = int(class_11.get(words[1])) + 1
        else:
            class_11[words[1]] = 1
    elif doc_class_val == 12:
        if class_12.has_key(words[1]):
            class_12[words[1]] = int(class_12.get(words[1])) + 1
        else:
            class_12[words[1]] = 1
    elif doc_class_val == 13:
        if class_13.has_key(words[1]):
            class_13[words[1]] = int(class_13.get(words[1])) + 1
        else:
            class_13[words[1]] = 1
    elif doc_class_val == 14:
        if class_14.has_key(words[1]):
            class_14[words[1]] = int(class_14.get(words[1])) + 1
        else:
            class_14[words[1]] = 1
    elif doc_class_val == 15:
        if class_15.has_key(words[1]):
            class_15[words[1]] = int(class_15.get(words[1])) + 1
        else:
            class_15[words[1]] = 1
    elif doc_class_val == 16:
        if class_16.has_key(words[1]):
            class_16[words[1]] = int(class_16.get(words[1])) + 1
        else:
            class_16[words[1]] = 1
    elif doc_class_val == 17:
        if class_17.has_key(words[1]):
            class_17[words[1]] = int(class_17.get(words[1])) + 1
        else:
            class_17[words[1]] = 1
    elif doc_class_val == 18:
        if class_18.has_key(words[1]):
            class_18[words[1]] = int(class_18.get(words[1])) + 1
        else:
            class_18[words[1]] = 1
    elif doc_class_val == 19:
        if class_19.has_key(words[1]):
            class_19[words[1]] = int(class_19.get(words[1])) + 1
        else:
            class_19[words[1]] = 1
    elif doc_class_val == 20:
        if class_20.has_key(words[1]):
            class_20[words[1]] = int(class_20.get(words[1])) + 1
        else:
            class_20[words[1]] = 1

#dic with test doc id and its corresponding class value
def build_test_class_dict(testlabel):
    f = open(test_label,'r')
    count = 1
    for line in f.readlines():
        word = line.strip("\n")
        test_doc_class[count] = int(word)
        count +=1
    #print len(test_doc_class)

#this is the method which calls this stupid implementation 
#which gets the number of docs in which the word occurs for every class 
def build_bernoulli_distr(label_file,vocab):
    f = open(label_file,'r')
    for line in f.readlines():
        words = line.split()
        if words[1] in vocab:
            add_word_to_appropriate_class_dict(words)

#construct test doc with docid as key and word n count as another dict
def build_doc_for_words(testfile):
    f = open(testfile,'r')
    for line in f.readlines():
        words = line.split()
        #dic = {}
        #dic[words[1]] = words[2]
        if doc_dict.has_key(words[0]):
            dict_of_words_count = doc_dict[words[0]]
            if dict_of_words_count.has_key(words[1]):
                #doc_dict[words[0]]doc_dict[words[0]].get(words[1])
                dict_of_words_count[words[1]] = int(dict_of_words_count[words[1]]) + int(words[2])
                doc_dict[words[0]] = dict_of_words_count
            else:
                dict_of_words_count[words[1]] = int(words[2])
                doc_dict[words[0]] = dict_of_words_count
        else:
            dict_of_words_count = {}
            dict_of_words_count[words[1]] = words[2]
            doc_dict[words[0]] = dict_of_words_count
    #print doc_dict

#method to get the count of every word in a doc
def return_count_term_in_doc(term,ele_in_doc):
    for i in ele_in_doc:
        if i.has_key(term):
            return int(i[term])
    return 0

#predict the class for the test doc
def predict_class(vocab_words,v):
    pred_dict = {}
    for doc_id,ele_in_doc in doc_dict.items():
        max_score = -99999999999
        final_class = 0
        #print "len of the doc is", len(ele_in_doc)
        for it,val in enumerate(cond_prob_class_list):
            c = str(it+1)
            prior_prob_for_class = prior_prob_dict[c]
            #score = math.log(prior_prob_for_class)
            score = prior_prob_for_class
            #print "prior prob for class", c,score
            for term in vocab_words:
                if ele_in_doc.has_key(term):
                    score += math.log(val[term])
                else:
                    score += math.log(1- val[term])
            if score > max_score:
                max_score = score
                final_class = it+1
        #print doc_id,max_score,final_class
        pred_dict[int(doc_id)] = int(final_class)
    return pred_dict

#acuuracy for the test data
def find_accuracy(d,val):
    no_of_right = 0
    no_of_wrong = 0
    no_of_docs = len(test_doc_class)
    for k in range(no_of_docs):
        #if int(d[int(k)]) != int(test_doc_class[int(k)]):
        #print "Value of pred is", d[k+1], "Value of test label is", test_doc_class[k+1]
        if int(d[k+1]) == int(test_doc_class[k+1]):
            no_of_right +=1
        else:
            no_of_wrong +=1
    #print "Sum of right and wrong is", no_of_wrong+no_of_right
    print "No of right values", no_of_right
    print "Accuracy for ",val, " ", "is", 100 * no_of_right/float(no_of_docs)

#cal conditional probability for each term in the vocabulary
def cal_conditional_term_in_vocab(vocab):
    for it,val in enumerate(class_list):
        no_of_docs = class_dict[str(it+1)]
        cond_prob_class = {}
        for term in vocab:
            if val.has_key(term):
                tf = int(val[term])
                cond_prob_class[term]= (tf+1)/float(no_of_docs+2)#math.log(tf+1)- math.log(no_of_docs+2)
            else:
                tf = 0
                #cond_prob_class[term] = 0
                cond_prob_class[term]= (tf+1)/float(no_of_docs+2)#math.log(tf+1)-math.log(no_of_docs+2)
        cond_prob_class_list[it] = cond_prob_class

if __name__ == "__main__":
    if len(sys.argv) == 5:
        train_file = sys.argv[1]
        label_file = sys.argv[2]
        test_file = sys.argv[3]
        test_label = sys.argv[4]
        cal_prior_prob_for_class(label_file)
        vocab_fre(train_file)
        build_test_class_dict(test_label)
        build_doc_for_words(test_file)
        Freq = sorted(word_dict.iteritems(),key=itemgetter(1),reverse=True)
        l = [100, 500, 1000, 2500, 5000, 7500, 10000, 12500, 25000, 50000]
        for v in l:
            p_list  = []
            vocab_words = []
            for i in range(v):
                tmp = Freq[i]
                vocab_words.append(tmp[0])
            #print(len(vocab_words))
            vocab_words = set(vocab_words)
            #print "new vacab len",len(vocab_words)
            #need to replace this stupid hardcoded logic
            class_1 = {}
            class_2 = {}
            class_3 = {}
            class_4 = {}
            class_5 = {}
            class_6 = {}
            class_7 = {}
            class_8 = {}
            class_9 = {}
            class_10 = {}
            class_11 = {}
            class_12 = {}
            class_13 = {}
            class_14 = {}
            class_15 = {}
            class_16 = {}
            class_17 = {}
            class_18 = {}
            class_19 = {}
            class_20 = {}
            class_list = [class_1,class_2,class_3,class_4,class_5,class_6,class_7,class_8,class_9,class_10,
                          class_11,class_12,class_13,class_14,class_15,class_16,class_17,class_18,class_19,class_20]
            build_bernoulli_distr(train_file,vocab_words)
            cond_prob_class_1 = {}
            cond_prob_class_2 = {}
            cond_prob_class_3 = {}
            cond_prob_class_4 = {}
            cond_prob_class_5 = {}
            cond_prob_class_6 = {}
            cond_prob_class_7 = {}
            cond_prob_class_8 = {}
            cond_prob_class_9 = {}
            cond_prob_class_10 = {}
            cond_prob_class_11 = {}
            cond_prob_class_12 = {}
            cond_prob_class_13 = {}
            cond_prob_class_14 = {}
            cond_prob_class_15 = {}
            cond_prob_class_16 = {}
            cond_prob_class_17 = {}
            cond_prob_class_18 = {}
            cond_prob_class_19 = {}
            cond_prob_class_20 = {}
            cond_prob_class_list = [cond_prob_class_1,cond_prob_class_2,cond_prob_class_3,cond_prob_class_4,cond_prob_class_5,cond_prob_class_6,cond_prob_class_7,cond_prob_class_8,cond_prob_class_9,cond_prob_class_10,
                          cond_prob_class_11,cond_prob_class_12,cond_prob_class_13,cond_prob_class_14,cond_prob_class_15,cond_prob_class_16,cond_prob_class_17,cond_prob_class_18,cond_prob_class_19,cond_prob_class_20]
            cal_conditional_term_in_vocab(vocab_words)
            pred_dict = predict_class(vocab_words,v)
            print pred_dict
            find_accuracy(pred_dict,v)
