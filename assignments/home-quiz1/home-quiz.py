import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

class homeQuiz:

    def extractData(self):
        # This function extracts the data from the file and seperates the reviews from the dataset.

        dataFrame = pd.read_csv("Movies_TV.txt", delimiter='\t')
        self.reviews = []
        for review in dataFrame.Review:
            self.reviews.append(review)
    
    def binaryStructureTfidf(self):
        vec = TfidfVectorizer(ngram_range=(1,3), max_df=100, min_df=10, max_features=1000, binary=True)
        X = vec.fit_transform(self.reviews)

        print(X.toarray())

    def freqStructureTfidf(self):
        vec = TfidfVectorizer(ngram_range=(1,3), max_df=100, min_df=10, max_features=1000)
        X = vec.fit_transform(self.reviews)

        print(X.toarray())

    def binaryStructure(self):
        vec = CountVectorizer(ngram_range=(1, 3), max_df=100,
                              min_df=10, max_features=1000, binary=True)
        X = vec.fit_transform(self.reviews)

        print(X.toarray())

    def freqStructure(self):
        vec = CountVectorizer(ngram_range=(1, 3), max_df=100, min_df=10, max_features=1000)
        X = vec.fit_transform(self.reviews)

        print(X.toarray())



if __name__ == "__main__":
    quizObj = homeQuiz()
    quizObj.extractData()
    quizObj.binaryStructureTfidf()
    quizObj.freqStructureTfidf()
    quizObj.freqStructure()
    quizObj.binaryStructure()
