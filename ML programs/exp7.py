import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,f1_score
from sklearn.naive_bayes import BernoulliNB,MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

train_data = fetch_20newsgroups(subset="train",remove=("headers","footers","quotes"))
test_data = fetch_20newsgroups(subset="test",remove=("headers","footers","quotes"))

X_train = train_data.data
y_train = train_data.target
X_test=test_data.data
y_test=test_data.target

vectorizer=CountVectorizer(stop_words="english")
X_train=vectorizer.fit_transform(X_train)
X_test=vectorizer.transform(X_test)

multi_model=MultinomialNB()
multi_model.fit(X_train,y_train)
multi_pred=multi_model.predict(X_test)
acc_multi=accuracy_score(y_test,multi_pred)
f1_multi=f1_score(y_test,multi_pred,average="weighted")


b_model=BernoulliNB()
b_model.fit(X_train,y_train)
b_pred=b_model.predict(X_test)
acc_b=accuracy_score(y_test,b_pred)
f1_b=f1_score(y_test,b_pred,average="weighted")
print("acc of multi is",acc_multi)
print("acc of bern is",acc_b)
