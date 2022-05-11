import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


class Model:

    def __init__(self):
        data = pd.read_csv('input_data.csv')
        self.X = data.iloc[:, [0, 1]]
        self.y = data.iloc[:, [2]]
        self.classifier = KNeighborsClassifier(20)
        self.classifier.fit(self.X.values, self.y.values)

    def predict(self, x):
        return self.classifier.predict(x)

