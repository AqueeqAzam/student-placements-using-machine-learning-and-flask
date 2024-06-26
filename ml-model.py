# copy this code and paste it in google colab or jupyter lab
import pandas as pd
import numpy as np


# Use the 'raw' URL to directly access the CSV data
df = pd.read_csv("https://raw.githubusercontent.com/AqueeqAzam/data-science-and-machine-learning-datasets/main/students-placement.csv")
df.head(3)

df.shape

# work same as head
df.sample(3)

x = df.drop(columns=['placed'])
y = df['placed']

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
svc = SVC()

accuracy_score(y_test, SVC(kernel='rbf').fit(x_train, y_train).predict(x_test))

svc = SVC(kernel = 'rbf')
svc.fit(x_train, y_train)

# run this command in other cell
import pickle
pickle.dump(svc, open('modle.pkl', 'wb'))
