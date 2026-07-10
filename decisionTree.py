# Potential libraries used
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LinearRegression


# Since I will be focusing more on home ai systems, https://github.com/alexa/massive will be a better fit 


# Takes in 1.0/data as that is where all the files are located
def load_dataset(data_dir="1.0/data"):
    data_path = Path(data_dir)
    jsonl_files = sorted(data_path.glob("*.jsonl"))

    if not jsonl_files:
        raise FileNotFoundError(f"No JSONL files found in {data_path.resolve()}")
    # Return a single dataframe and concat(stack) all the jsonl files into one dataframe
    dataframes = [pd.read_json(file_path, lines=True) for file_path in jsonl_files]
    return pd.concat(dataframes, ignore_index=True)


def main():
    # Step One: Use the dataset and explore different features and labels
    # Convert all JSONL files to one dataframe and check columns and first few rows
    df = load_dataset()
    print("STEP ONE")
    print(df.columns)
    print(df.head())

    # Step Two: Preprocess by reviewing the data 
    print("STEP TWO")
    df.info()
    df.describe()
    print(df.shape)

    # Split on utterance text and intent label
    X = df["utt"]  # Utterances (text)
    y = df["intent"]  # Intent labels
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    
    # Vectorize the text using TF-IDF
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Step Three: Train a text classifier on utterances
    model = DecisionTreeClassifier()
    model.fit(X_train_vec, y_train)
    
    print("STEP THREE")
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    # Step Four: Evaluate overall accuracy and accuracy by locale-specific groups
    # Test accuracy is calculated as 0.282
    print(f"Classifier trained. Test accuracy: {accuracy:.3f}")    

    # Step Five: Improve the data and calculate fairness on a 0-10 scale
    
    # Final step: Predict and compare fairness across locales

if __name__ == "__main__":
    main()