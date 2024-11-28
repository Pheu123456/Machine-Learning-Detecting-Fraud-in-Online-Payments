import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
import seaborn as sns # Seaborn is a useful library for Data Visualisation
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import plot_tree
from sklearn.tree import export_graphviz 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
df = pd.read_csv('F:/uit/AI/Finallllll/onlinefraud.csv')
d = {'CASH_OUT': 0, 'PAYMENT': 1, 'CASH_IN': 2, 'TRANSFER': 3, 'DEBIT': 4}
df['type'] = df['type'].map(d)
df = df.drop(columns=['nameOrig']) #name
df = df.drop(columns=['nameDest']) #name


# slit and train
X= df.drop(columns='isFraud')
y= df['isFraud']
df.dropna(inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


clf = DecisionTreeClassifier(max_depth=10, min_samples_split=10, random_state=42)


clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)

print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
print(classification_report(y_test, y_pred))
plt.figure(figsize=(12,12)) 
tree.plot_tree(clf)
plt.show()
