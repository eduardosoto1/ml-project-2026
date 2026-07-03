# Potential libraries used
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier


# Other potential Data Set: https://archive.ics.uci.edu/dataset/518/speaker+accent+recognition
# Other potential data set: https://accent.gmu.edu/
# Since I will be focusing more on home ai systems, https://github.com/alexa/massive will be a better fit 


def main():
    # Step One: Use Data Set
    import pandas as pd 
    # Convert the json file to a dataframe and check columns and first few rows
    df = pd.read_json("1.0/data/en-US.jsonl", lines=True)
    print(df.columns)
    print(df.head())
    

    # Step Two: Explore different features (x) and samples per accent

    # Step Three: Preprocess by cleaning data and splitting features and labels

    # Step Four: Train classifer model on voice features

    # Step Five: Evaluate overall accuracy then accuracy by accent group


    # Step Six: Improve data 
    
    # Step Seven: Predict the accent from voice features and check fairness across accents

if __name__ == "__main__":
    main()