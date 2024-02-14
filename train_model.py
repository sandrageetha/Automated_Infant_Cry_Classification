from feature_extraction import extract_features, load_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pickle

# Load data
audio_paths, labels = load_data()

# Extract features for each audio file
features = []
for path in audio_paths:
    features.append(extract_features(path))

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(labels)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(np.array(features), y_encoded, test_size=0.2, random_state=42)

# Train KNN
print("Training KNN...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
print("KNN Score: ", knn.score(X_test, y_test))

# Train SVM
print("Training SVM...")
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)
print("SVM Score: ", svm.score(X_test, y_test))

# Train Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
print("Random Forest Score: ", rf.score(X_test, y_test))

# Save models
pickle.dump(knn, open('models/knn_model.pkl', 'wb'))
pickle.dump(svm, open('models/svm_model.pkl', 'wb'))
pickle.dump(rf, open('models/rf_model.pkl', 'wb'))
