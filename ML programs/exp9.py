import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report


data = load_iris()
X = data.data
y = data.target

print("Dataset shape:", X.shape)
print("Features:", data.feature_names)
print("Classes:", data.target_names)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier(criterion="entropy",max_depth=3,random_state=42)

model.fit(X_train, y_train)

model_pred= model.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, model_pred))

print("\nClassification Report:")
print(classification_report(y_test, model_pred))

print("\nFeature Importance:")

for feature,importance in zip(data.feature_names,model.feature_importances_):
    print(feature,"",importance)
plt.figure(figsize=(14, 8))

plot_tree(
    model,
    feature_names=data.feature_names,
    class_names=data.target_names,

)

plt.title("Decision Tree - Iris Dataset")
plt.show()
plt.bar(data.feature_names,model.feature_importances_)
plt.xlabel("features")
plt.ylabel("importance")
plt.show()
