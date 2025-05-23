import joblib
import pandas as pd
from nltk.stem import PorterStemmer
import re

model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

df = pd.read_csv("IMDB Dataset.csv")

with open("stopwords.txt", "r") as f:
    stopwords = set(line.strip() for line in f)

stemmer = PorterStemmer()

def clear_screen():
    print("\n" * 50)


def preprocess_text(text):
    text = re.sub(r"<.*?>", " ", text)
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\d+", "", text)
    words = text.split()
    words = [word for word in words if word not in stopwords]
    words = [stemmer.stem(word) for word in words]
    return " ".join(words)


def option1():
    try:
        index = int(input("Enter a review number (0 to 49999): "))
        if 0 <= index < len(df):
            original_review = df["review"].iloc[index]
            actual_sentiment = df["sentiment"].iloc[index]

            # Preprocess and predict
            cleaned_review = preprocess_text(original_review)
            vectorized_review = vectorizer.transform([cleaned_review])
            predicted_sentiment = model.predict(vectorized_review)[0]

            clear_screen()
            print("Review:")
            print("-" * 60)
            print(original_review)
            print("-" * 60)
            print("Actual Sentiment:   ", actual_sentiment.capitalize())
            print("Predicted Sentiment:", predicted_sentiment.capitalize())
        else:
            print("Invalid number. Please enter a number between 0 and 49999.")
    except ValueError:
        print("Please enter a valid number.")

    input("\nPress Enter to return to the main menu...")


def option2():
    user_input = input("\nType your review:\n")
    
    if not user_input.strip():
        print("Review cannot be empty.")
        return

    # Preprocess, vectorize, and predict
    cleaned_review = preprocess_text(user_input)
    vectorized_review = vectorizer.transform([cleaned_review])
    predicted_sentiment = model.predict(vectorized_review)[0]

    clear_screen()
    print("Your Review:")
    print("-" * 60)
    print(user_input)
    print("-" * 60)
    print("Predicted Sentiment:", predicted_sentiment.capitalize())

    input("\nPress Enter to return to the main menu...")



while True:
    clear_screen()
    print("AI Movie Review Sentiment Analyzer")
    print("-----------------------------------")
    print("1. Test sentiment of a review from the IMDb dataset")
    print("2. Enter your own review")
    print("3. Exit")
    choice = input("\nChoose an option (1/2/3): ").strip()

    if choice == "1":
        option1()
    elif choice == "2":
        option2()
    elif choice == "3":
        print("Goodbye!")
        break
    else:
        print("Invalid option. Please try again.")
