import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes    import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics        import accuracy_score, classification_report

# (a)
print("\n-------------------------")
print("(a)")
print("-------------------------\n")

# Download NLTK resources (only required on first run)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# 1. Load the CSV file
df = pd.read_csv("./Text_Classification/L06_NLP_LLM_emails.csv")  # Check file path if necessary
print("Sample data:\n", df.head())

# 2. Define text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize
    tokens = text.split()
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    # Join tokens back into a single string
    return ' '.join(tokens)

# 3. Apply preprocessing to all documents
df['processed_text'] = df['text'].apply(preprocess_text)

# 4. Create Document-Term Matrix (DTM)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['processed_text'])

# Labels (Spam or not)
y = df['spam']

print("DTM shape:", X.shape)  # Example: (n_documents, n_features)



# (b)
print("\n-------------------------")
print("(b)")
print("-------------------------\n")

# 1. Load and preprocess
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return ' '.join(tokens)

df['processed_text'] = df['text'].apply(preprocess_text)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['processed_text'])
y = df['spam']

# 2. Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Train Naïve Bayes
nb_clf = MultinomialNB()
nb_clf.fit(X_train, y_train)
y_pred_nb = nb_clf.predict(X_test)
acc_nb = accuracy_score(y_test, y_pred_nb) * 100

# 4. Train MLP
mlp_clf = MLPClassifier(
    hidden_layer_sizes=(100,),
    activation='relu',
    solver='adam',
    max_iter=300,
    random_state=42
)
mlp_clf.fit(X_train, y_train)
y_pred_mlp = mlp_clf.predict(X_test)
acc_mlp = accuracy_score(y_test, y_pred_mlp) * 100

# 5. Print the results
print(f"Naïve Bayes Accuracy: {acc_nb:.2f}%")
print(classification_report(y_test, y_pred_nb, target_names=['Ham','Spam']))
print(f"MLP Accuracy:        {acc_mlp:.2f}%")
print(classification_report(y_test, y_pred_mlp, target_names=['Ham','Spam']))

# 6. Choose model and simple justification
if acc_nb >= acc_mlp:
    print("\n=> Recommendation: Use Naïve Bayes for spam filtering.")
    print("   Justification: Comparable or higher accuracy, far faster training/inference,")
    print("   and simpler to deploy in production.")
else:
    print("\n=> Recommendation: Use MLP for spam filtering.")
    print("   Justification: Higher accuracy on this dataset, better at capturing complex")
    print("   feature interactions, though at the cost of longer training time.")
