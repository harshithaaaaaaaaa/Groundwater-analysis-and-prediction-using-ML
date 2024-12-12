import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
import joblib

# Load the data
data = pd.read_csv("data.csv")
print(data.head())  # Display the first few rows

# Visualize the target variable distribution
plt.figure(figsize=(10, 6))
sb.countplot(x='Situation', data=data, palette='bright')
plt.title('Distribution of Groundwater Situation')
plt.show()

# Select features and target variable
X = data.iloc[:, [4, 5, 6, 9, 10, 11]]
Y = data.iloc[:, 12]

# Encode the target variable
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

# Split the data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)

# Scale the features
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Feature selection
selector = SelectKBest(f_classif, k=6)  # Select all 6 features
X_train_selected = selector.fit_transform(X_train, Y_train)
X_test_selected = selector.transform(X_test)

# Train the logistic regression model
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train_selected, Y_train)

# Calculate probabilities
y_prob = classifier.predict_proba(X_test_selected)[:, 1]

# Set custom threshold
threshold = 0.5
y_pred_custom_threshold = (y_prob >= threshold).astype(int)

# Evaluate the model
cm = confusion_matrix(Y_test, y_pred_custom_threshold)
score = accuracy_score(Y_test, y_pred_custom_threshold)
print("Confusion Matrix:\n", cm)
print("Accuracy Score:", score)

# Encode 'Situation' with one-hot encoding
Availability = pd.get_dummies(data['Situation'], drop_first=True)
data.drop(['Situation'], axis=1, inplace=True)
data1 = pd.concat([data, Availability], axis=1)
X = data1.iloc[:, [4, 5, 6, 9, 10, 11]]
Y = data1.iloc[:, 12]

# Split the data again with a different test size
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.55, random_state=0)

# Scale the features
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Feature selection
X_train_selected = selector.fit_transform(X_train, Y_train)
X_test_selected = selector.transform(X_test)

# Train the logistic regression model again
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train_selected, Y_train)

# Calculate probabilities again
y_prob = classifier.predict_proba(X_test_selected)[:, 1]

# Make predictions based on the same custom threshold
y_pred_custom_threshold = (y_prob >= threshold).astype(int)

# Evaluate the new model
cm = confusion_matrix(Y_test, y_pred_custom_threshold)
new_score = accuracy_score(Y_test, y_pred_custom_threshold)
print("New Confusion Matrix:\n", cm)
print("New Accuracy Score:", new_score)

# Save the model, scaler, and selector
joblib.dump(classifier, 'groundwater_model.joblib')
joblib.dump(sc, 'groundwater_scaler.joblib')
joblib.dump(selector, 'groundwater_selector.joblib')

print("Model, scaler, and selector have been saved.")