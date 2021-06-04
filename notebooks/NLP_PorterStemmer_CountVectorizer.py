import nltk
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score


def solution():
    # Read the file as a pandas data-frame.
    df = pd.read_csv('res/Eopinions.csv')
    print(df.head())

    # Perform Label Encoding on ‘class’ column.
    le = LabelEncoder()
    df['class'] = le.fit_transform(df['class'])
    print(df['class'].value_counts())

    # Text preprocessing
    nltk.download('stopwords')
    corpus = []
    ps = PorterStemmer()
    for i in range(len(df)):
        # removing special characters
        text = re.sub('[^a-zA-Z]', ' ', df['text'][i]).lower().split()
        # Stemming and removing stop words
        text = [ps.stem(word) for word in text if word not in
                set(stopwords.words('english'))]
        # Joining all the cleaned words
        text = ' '.join(text)
        # add the cleaned sentence to a list
        corpus.append(text)

    # Vectorize the text using CountVectorizer
    cv = CountVectorizer(max_features=120)
    X = cv.fit_transform(corpus).toarray()
    y = df.iloc[:, 0].values

    # Split the dataset into 2 parts
    X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, test_size=0.20, random_state=42)

    # Train your machine learning algorithm for classification
    clf = SVC()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Now test the model on the Test data and evaluate the Performance.
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
