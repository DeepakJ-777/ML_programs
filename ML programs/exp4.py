import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix


data = load_breast_cancer()

X = data.data
y = data.target

print("Dataset shape:", X.shape)
print("Features:", data.feature_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mle_model=LogisticRegression(penalty=None,max_iter=1000)
mle_model.fit(X_train,y_train)
mle_pred=mle_model.predict(X_test)
mle_prob=mle_model.predict_proba(X_test)

print("mle")
print("acc is",accuracy_score(y_test,mle_pred))
print("log loss is",log_loss(y_test,mle_prob))
print("coefficients are",mle_model.coef_)

l2_model=LogisticRegression(penalty="l2",C=1.0,max_iter=10000)
l2_model.fit(X_train,y_train)
l2_pred=mle_model.predict(X_test)
l2_prob=mle_model.predict_proba(X_test)

print("l2")
print("acc is",accuracy_score(y_test,l2_pred))
print("log loss is",log_loss(y_test,l2_prob))
print("coefficients are",l2_model.coef_)

l1_model=LogisticRegression(penalty="l1",solver="liblinear",C=1.0,max_iter=10000)
l1_model.fit(X_train,y_train)
l1_pred=mle_model.predict(X_test)
l1_prob=mle_model.predict_proba(X_test)

print("l1")
print("acc is",accuracy_score(y_test,l1_pred))
print("log loss is",log_loss(y_test,l1_prob))
print("coefficients are",l1_model.coef_)

models = ["MLE", "MAP-L2", "MAP-L1"]

predictions = [mle_pred, l2_pred, l1_pred]
probabilities = [mle_prob, l2_prob, l1_prob]
trained_models = [mle_model, l2_model, l1_model]

print("\nModel Comparison")
print("Model\t\tAccuracy\tLog Loss")

for i in range(3):
    print(
        models[i], "\t",
        round(accuracy_score(y_test, predictions[i]), 4), "\t\t",
        round(log_loss(y_test, probabilities[i]), 4)
    )

# Compare parameter estimates
print("\nNumber of non-zero coefficients:")

for i in range(3):
    count = np.count_nonzero(trained_models[i].coef_)
    print(models[i], ":", count)

# Plot accuracy comparison
accuracies = [
    accuracy_score(y_test, mle_pred),
    accuracy_score(y_test, l2_pred),
    accuracy_score(y_test, l1_pred)]

import matplotlib.pyplot as plt

plt.bar(models, accuracies)
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.title("MLE vs MAP Logistic Regression")
plt.grid(axis="y")
plt.show()
