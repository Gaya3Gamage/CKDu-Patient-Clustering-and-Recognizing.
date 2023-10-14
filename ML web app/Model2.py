import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load the csv file

df = pd.read_csv("Urine_top_CKDuWS.csv")

print(df.head())

# Select independent and dependent variable
X = df[["Rb","Bi","Hg","U","Al","Ag","Ga","Li","Pb","Age"]]
y = df["Group"]

# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=65)



# Instantiate the model
classifier = RandomForestClassifier()

# Fit the model
classifier.fit(X_train, y_train)

# Fit the model
classifier.fit(X_train, y_train)

# Make pickle file of our model
pickle.dump(classifier, open("model2.pkl", "wb"))