import pandas as pd
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.model_selection import cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report,confusion_matrix

#load dataset
df = pd.read_csv("fake reviews dataset.csv")
df['text_'] = df['text_'].astype(str)

#check data
'''
print(df.isnull().sum())
print(df["label"].value_counts())
print(df["label"].unique())
'''

train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["label"])


#text classification pipeline Stage 1
nb_pipeline = Pipeline([
    ("tfidf",TfidfVectorizer(stop_words="english")),
    ("nb",MultinomialNB())
])

X_text_train = train_df['text_']
y_train = train_df["label"]


# Generate out-of-fold probabilities for training data
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_probs = cross_val_predict(
    nb_pipeline,
    X_text_train,
    y_train,
    cv=cv,
    method="predict_proba"
)


# Replace "CG" below with the actual fake-label value in your dataset.
fake_label = "CG"
classes = nb_pipeline.fit(X_text_train, y_train).classes_
fake_idx = list(classes).index(fake_label)

train_meta = train_df[["category", "rating"]].copy()
train_meta["nb_fake_probability"] = oof_probs[:, fake_idx]

# Stage 2: category + rating + NB probability
preprocessor = ColumnTransformer([
    ("category", OneHotEncoder(handle_unknown="ignore"),
     ["category"]),
    ("rating", SimpleImputer(strategy="median"),
     ["rating"]),
    ("nb_probability", "passthrough",
     ["nb_fake_probability"])
])

lr_pipeline = Pipeline([
    ("features", preprocessor),
    ("lr", LogisticRegression(max_iter=1000))
])

lr_pipeline.fit(train_meta, y_train)

# Prepare test features using NB trained on training data
test_probs = nb_pipeline.predict_proba(test_df["text_"])
test_meta = test_df[["category", "rating"]].copy()
test_meta["nb_fake_probability"] = test_probs[:, fake_idx]

final_pred = lr_pipeline.predict(test_meta)

print(classification_report(test_df["label"], final_pred))



# ---------- Final Prediction Results (Both)----------

#review details
category = input("Enter product category: ")
rating = float(input("Enter rating (1-5): "))
review_text = input("Enter review text: ")

#Naive Bayes predicts probability from text
nb_probabilities = nb_pipeline.predict_proba([review_text])
fake_probability = nb_probabilities[0, fake_idx]

#Prepare input for Logistic Regression
new_review = pd.DataFrame({
    "category": [category],
    "rating": [rating],
    "nb_fake_probability": [fake_probability]
})

#Final prediction
prediction = lr_pipeline.predict(new_review)[0]

#result
print("\n----- Review Analysis -----")
print("Category:", category)
print("Rating:", rating)
print("Naive Bayes fake probability:",
      round(fake_probability * 100, 2), "%")

if prediction == fake_label:
    print("Final Prediction: FAKE REVIEW")
else:
    print("Final Prediction: GENUINE REVIEW")


#---------- Bayes Prediction --------

# Input review
review_text = input("Enter review text: ")

# Naive Bayes prediction
prediction = nb_pipeline.predict([review_text])[0]

# Probability of each class
probabilities = nb_pipeline.predict_proba([review_text])[0]
classes = nb_pipeline.classes_

print("\nNaive Bayes Prediction:", prediction)

for label, probability in zip(classes, probabilities):
    print(f"{label}: {probability * 100:.2f}%")
