#  Multivariate Bernoulli and Multinomial event based NaiveBased Document Classification

Goal of this project is to implement NaiveBased Classifier without using any Machine Learning libraries

In this project, I implemented NaiveBased classifier on a very popular news data set called 20newsgroups data set.
The data is composed of six files, three of them contain the test data while the other
three have the training data. Each row of the train.data and test.data files contain the data listed
as (docId, wordId, count). The train.label and test.label files contain the labels for each document.
 
https://archive.ics.uci.edu/ml/machine-learning-databases/20newsgroups-mld/20newsgroups.data.html

Here are the 20 classes from this data set:

1. alt.atheism 1
2. comp.graphics 2
3. comp.os.ms-windows.misc 3
4. comp.sys.ibm.pc.hardware 4
5. comp.sys.mac.hardware 5
6. comp.windows.x 6
7. misc.forsale 7
8. rec.autos 8
9. rec.motorcycles 9
10. rec.sport.baseball 10
11. rec.sport.hockey 11
12. sci.crypt 12
13. sci.electronics 13
14. sci.med 14
15. sci.space 15
16. soc.religion.christian 16
17. talk.politics.guns 17
18. talk.politics.mideast 18
19. talk.politics.misc 19
20. talk.religion.misc 20

I implemented both the models based on this configurations:

1. Create a word-frequency list across the training documents and sort it in descending order
from highest frequency to lowest frequency. We will be working with vocabulary sizes of |V | ∈
top{100, 500, 1000, 2500, 5000, 7500, 10000, 12500, 25000, 50000, All}, where “All” is using the
complete vocabulary.

2. Fit the multivariate Bernoulli model to the training data and evaluate the accuracy on the
test set. Keep in mind to restrict the vocabulary to the selected value of |V | for both the
training and test sets.

3. Fit the multivariate event model to the training data and evaluate the accuracy on the test
set. Keep in mind to restrict the vocabulary to the selected value of |V | for both the training
and test sets.

4. Use a simple smoothing model that assigns a default frequency of 1 to each word from the
vocabulary for both models(Laplace smoothing)
