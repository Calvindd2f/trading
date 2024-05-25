# Load preprocessed data
data = pd.read_csv('historical_data.csv')

# Define features and labels
features = ['price_change', 'volume_change', 'ma_10', 'ma_50', 'ma_200', 'ma_diff']
X = data[features]
y = data['label']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Save the trained model
joblib.dump(model, 'pump_dump_model.pkl')
