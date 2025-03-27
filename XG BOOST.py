import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load your dataset
df = pd.read_csv("network_traffic.csv")

# 1. Filter for DoS and Benign traffic only
df = df[df['Attack'].isin(['DoS', 'Benign'])]

# 2. Drop irrelevant columns
df.drop(columns=['IPV4_SRC_ADDR', 'IPV4_DST_ADDR'], inplace=True)

# 3. Encode target: 1 = DoS, 0 = Benign
df['Label'] = df['Attack'].apply(lambda x: 1 if x == 'DoS' else 0)

# 4. Select features
features = ['L4_SRC_PORT', 'L4_DST_PORT', 'PROTOCOL', 'L7_PROTO',
            'IN_BYTES', 'OUT_BYTES', 'IN_PKTS', 'OUT_PKTS',
            'TCP_FLAGS', 'FLOW_DURATION_MILLISECONDS']

X = df[features]
y = df['Label']

# 5. Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 6. Display result summary
print("Preprocessing Complete.")
print("Shape of input data:", X_scaled.shape)
print("\nClass Distribution:")
print(df['Label'].value_counts().rename(index={0: 'Benign', 1: 'DoS'}))

from sklearn.model_selection import train_test_split

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import time

# ✅ Step 1: Train-test split (assuming X_scaled and y are already prepared from preprocessing)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# ✅ Step 2: Define the model
xgb = XGBClassifier(eval_metric='logloss', random_state=42)

# ✅ Step 3: Train the model
start_train = time.time()
xgb.fit(X_train, y_train)
end_train = time.time()

# ✅ Step 4: Predict
start_test = time.time()
y_pred = xgb.predict(X_test)
y_prob = xgb.predict_proba(X_test)[:, 1]  # Needed for AUROC
end_test = time.time()

# ✅ Step 5: Evaluate
print("📌 XGBoost Results:")
print("Training Time: {:.2f} sec".format(end_train - start_train))
print("Testing Time : {:.2f} sec".format(end_test - start_test))
print("Accuracy     :", accuracy_score(y_test, y_pred))
print("Precision    :", precision_score(y_test, y_pred))
print("Recall       :", recall_score(y_test, y_pred))
print("F1 Score     :", f1_score(y_test, y_pred))
print("AUROC        :", roc_auc_score(y_test, y_prob))
