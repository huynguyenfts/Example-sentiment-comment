import pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score

class DataSource(object):

    def _load_raw_data(self, filename, is_train=True):

        a = []
        b = []

        regex = 'train_'
        if not is_train:
            regex = 'test_'

        with open(filename, 'r') as file:
            for line in file:
                if regex in line:
                    b.append(a)
                    a = [line]
                elif line != '\n':
                    a.append(line)

        b.append(a)

        return b[1:]

    def _create_row(self, sample, is_train=True):

        d = {}
        d['id'] = sample[0].replace('\n', '')
        review = ""

        if is_train:
            for clause in sample[1:-1]:
                review += clause.replace('\n', '').strip()
            d['label'] = int(sample[-1].replace('\n', ''))
        else:
            for clause in sample[1:]:
                review += clause.replace('\n', '').strip()

        d['review'] = review

        return d

    def load_data(self, filename, is_train=True):

        raw_data = self._load_raw_data(filename, is_train)
        lst = []

        for row in raw_data:
            lst.append(self._create_row(row, is_train))

        return lst


ds = DataSource()
train_data = pd.DataFrame(ds.load_data('train.crash'))
test_data = pd.DataFrame(ds.load_data('test.crash', is_train=False))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

vectorizer = TfidfVectorizer(max_features=100000, ngram_range=(1, 3))

x_train, x_val, y_train, y_val = train_test_split(train_data.review, train_data.label, test_size=0.3,
                                                  random_state=42)

vectorizer.fit(x_train)
x_tfidf_train = vectorizer.transform(x_train)
x_tfidf_val = vectorizer.transform(x_val)



model = LogisticRegression()
print("Train model.......")
sentiment_fit = model.fit(x_tfidf_train, y_train)
print("Predict ......")
y_pred = sentiment_fit.predict(x_tfidf_val)
accuracy = accuracy_score(y_val, y_pred)
print("accuracy score: {0:.2f}%".format(accuracy * 100))

x_tfidf_test = vectorizer.transform(test_data.review)
y_predict = sentiment_fit.predict(x_tfidf_test)
test_data['label'] = y_predict
test_data[['id', 'label']].to_csv('sample.csv', index=False)