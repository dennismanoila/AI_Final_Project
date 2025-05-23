import pandas as pd
import re
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib


#loading dataset
df = pd.read_csv("IMDB Dataset.csv")

#loading stopwords
with open("stopwords.txt", "r") as f:
    stopwords = set(line.strip() for line in f)

#preprocessing text function
stemmer = PorterStemmer()

def preprocess_text(text):
    text = re.sub(r"<.*?>", " ", text)  # remove HTML
    text = text.lower()  # lowercase
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\d+", "", text)  # remove digits
    words = text.split()
    words = [word for word in words if word not in stopwords] #selecting non stopwords
    words = [stemmer.stem(word) for word in words] #stemming ("beauty", "beautiful" -> "beauti")
    return " ".join(words)

#preprocessing all reviews
df["clean_review"] = df["review"].apply(preprocess_text)

#testing
#print(df[["review", "clean_review", "sentiment"]].head())

#tf idf vectorization

vectorizer = TfidfVectorizer(max_features=5000) #creating vectorizer (only 5000 words for keeping the most important 5k words)

X = vectorizer.fit_transform(df["clean_review"]) #the tf-idf matrix
Y = df["sentiment"] #the labels

#print("TF-IDF matrix shape:", X.shape)

#train/test split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

#train the logistic regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

#evaluate model
Y_pred = model.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred)

# print("Model Accuracy: ", accuracy)
# print("\nClassification Report:\n", classification_report(Y_test, Y_pred))

#save model and vectorizer
joblib.dump(model, "logistic_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")