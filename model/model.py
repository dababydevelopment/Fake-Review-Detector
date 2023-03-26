from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from model.preprocess import preprocess, preprocess_text
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

from joblib import dump, load

vectorizer = CountVectorizer()
df=pd.read_csv('model/dataset.csv')
X = df['text']
y = df['label']
X = preprocess(X)
vectorizer = vectorizer.fit(X)
X = vectorizer.transform(X)


def predict(text):
    # first, load the model.joblib as clf
    clf = load("model.joblib")
    # second, vectorize text
    text=preprocess_text(text)
    print("yay " + text)
    vectorized_text = vectorizer.transform([text])
    print(vectorized_text.toarray())
    # third, make and return the prediction
    return (clf.predict(vectorized_text)[0])
def main3():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)


    #neigh = KNeighborsClassifier(n_neighbors=3)
    #neigh.fit(X_train, y_train)

    #clf = svm.SVC()
    #clf.fit(X_train, y_train)


    #clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
    #                    hidden_layer_sizes=(5, 2), random_state=1)

    #clf.fit(X_train, y_train)


    clf = LogisticRegression(random_state=0, max_iter=100000000000000).fit(X_train, y_train)
    dump(clf, "model.joblib")
    
    y_pred = clf.predict(X_test)

    print(classification_report(y_test, y_pred))

    #vect = vectorizer.transform(["Very good product. I can't prove this for certain, but I think it cured my cancer. I feel like I'm 17 again."])
    #print(clf.predict(vect))

if __name__=="__main__":
    main3()