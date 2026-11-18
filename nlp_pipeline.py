import re
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score

# 1. Load Phase 1 Dataset
df = pd.read_csv("logistics_support_tickets.csv")

# ---------------------------------------------------------
# STEP 1: DETERMINISTIC ENTITY EXTRACTION (NER / REGEX)
# ---------------------------------------------------------
known_carriers = ["CN Rail", "Canada Post", "Challenger Motor Freight", "FedEx Express"]
known_hubs = [
    "Toronto Hub",
    "Mississauga Hub",
    "Montreal Hub",
    "Vancouver Hub",
    "Chicago Cross-Border Hub",
]

def extract_shipment_id(text):
    match = re.search(r"SHP-\d{6}", str(text))
    return match.group(0) if match else "UNKNOWN"

def extract_entity_from_list(text, entity_list):
    for item in entity_list:
        if item.lower() in str(text).lower():
            return item
    return "UNKNOWN"

df["extracted_shipment_id"] = df["raw_email_text"].apply(extract_shipment_id)
df["extracted_carrier"] = df["raw_email_text"].apply(lambda x: extract_entity_from_list(x, known_carriers))
df["extracted_hub"] = df["raw_email_text"].apply(lambda x: extract_entity_from_list(x, known_hubs))

# Audit Extraction Accuracy against Ground Truth
shp_acc = (df["extracted_shipment_id"] == df["true_shipment_id"]).mean() * 100
carrier_acc = (df["extracted_carrier"] == df["true_carrier"]).mean() * 100

print("=== STEP 1: ENTITY EXTRACTION AUDIT ===")
print(f"Shipment ID Extraction Accuracy: {shp_acc:.2f}%")
print(f"Carrier Extraction Accuracy:     {carrier_acc:.2f}%\n")

# ---------------------------------------------------------
# STEP 2: NLP TEXT CLEANING
# ---------------------------------------------------------
def clean_ticket_text(text):
    text = str(text).lower()
    # Replace SHP-XXXXXX with a generic token so model doesn't overfit to specific IDs
    text = re.sub(r"shp-\d{6}", "shipment_token", text)
    # Keep letters and spaces only
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

df["clean_text"] = df["raw_email_text"].apply(clean_ticket_text)

# ---------------------------------------------------------
# STEP 3: TRAIN DEPARTMENT ROUTING MODEL (4-CLASS)
# ---------------------------------------------------------
X = df["clean_text"]
y_dept = df["department_category"]
y_urg = df["urgency_level"]

X_train, X_test, y_dept_train, y_dept_test, y_urg_train, y_urg_test = train_test_split(
    X, y_dept, y_urg, test_size=0.20, random_state=42, stratify=y_dept
)

dept_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=1500)),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

dept_pipeline.fit(X_train, y_dept_train)
dept_preds = dept_pipeline.predict(X_test)

print("=== STEP 2: DEPARTMENT ROUTING PERFORMANCE ===")
print(f"Accuracy: {accuracy_score(y_dept_test, dept_preds):.4f}")
print(classification_report(y_dept_test, dept_preds))

# ---------------------------------------------------------
# STEP 4: TRAIN URGENCY TRIAGE MODEL (HIGH / MEDIUM / LOW)
# ---------------------------------------------------------
urg_pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=1500)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
])

urg_pipeline.fit(X_train, y_urg_train)
urg_preds = urg_pipeline.predict(X_test)

print("=== STEP 3: URGENCY TRIAGE PERFORMANCE ===")
print(f"Accuracy: {accuracy_score(y_urg_test, urg_preds):.4f}")
print(classification_report(y_urg_test, urg_preds))

# ---------------------------------------------------------
# STEP 5: EXPORT ARTIFACTS & ENRICHED DATASET
# ---------------------------------------------------------
joblib.dump(dept_pipeline, "department_router_model.joblib")
joblib.dump(urg_pipeline, "urgency_triage_model.joblib")
df.to_csv("enriched_logistics_tickets.csv", index=False)

print("✅ Phase 2 Complete! Saved trained pipelines (.joblib) and enriched_logistics_tickets.csv")