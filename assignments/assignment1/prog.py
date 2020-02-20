import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from string import punctuation as punc
from nltk import ngrams
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer


wl = WordNetLemmatizer()
ps = PorterStemmer()
#Seprating dataset into 4 Columns.

df = pd.read_csv("Movies_TV.txt", delimiter="\t")

#Removing whitespace from reviews.
reviews = []
for review in df.Review:
    reviews.append(review.split(" "))


#Removing Stop Words, Puncuations and truing all words to lower case
tempReviews = []
for review in reviews:
    tempReview = []
    for word in review:
        if (word not in ENGLISH_STOP_WORDS) and (word not in punc):
            word = word.lower()
            word = ps.stem(word)
            word = wl.lemmatize(word, 'a')
            word = wl.lemmatize(word, 'v')
            word = wl.lemmatize(word, 'n')
            tempReview.append(word)

    tempReviews.append(tempReview)

PreporcessedReviews = tempReviews


unigrams = list(ngrams(PreporcessedReviews[0], 1))
bigrams = list(ngrams(PreporcessedReviews[0], 2))
trigrams = list(ngrams(PreporcessedReviews[0], 3))
print("Unigrams    :", unigrams, "\n")
print("Bigrams     :", bigrams, "\n")
print("Trigrams    :", trigrams, "\n")

print("Number of Unigrams    :",  len(unigrams))
print("Numbers of Bigrams    :",  len(bigrams))
print("Numbers of Trigrams   :",  len(trigrams))

#Counting Tokens
print("Total tokens in each review: \n")

reviewNumber = 1
for review in reviews:
    print("Review ", reviewNumber, ":", len(review))
    reviewNumber += 1


words = []
for review in reviews:
    for word in review:
        words.append(word)

preprocessedWords = []
for review in PreporcessedReviews:
    for word in review:
        preprocessedWords.append(word)

print("Vcabulary Before preprocessing: \n")

cv = CountVectorizer(words)
count_vector = cv.fit_transform(words)
print(cv.vocabulary_)

print("Vcabulary After preprocessing: \n")

cv = CountVectorizer(preprocessedWords)
count_vector = cv.fit_transform(preprocessedWords)
print(cv.vocabulary_)

#finding the average lenght of a review.
totalReviews = 0
totalWords = 0
for review in reviews:
    totalReviews += 1
    totalWords += len(review)

print('\nAverage lenght of a review:', totalWords/totalReviews, 'words')

totalReviews = 0
for review in reviews:
    totalReviews += 1
    totalWords = 0
    totalWordLen = 0
    for word in review:
        totalWords += 1
        totalWordLen += len(word)

    print("Average Length of Tokens in Review",
          totalReviews, ":", totalWordLen/totalWords)
