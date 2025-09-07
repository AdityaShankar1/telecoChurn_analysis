import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import shap

# Optional: make plots look cleaner
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

df = pd.read_csv("telecom_churn.csv")

# Peek at the data
print(df.shape)
print(df.columns)
df.head()

# Basic info
df.info()

# Summary stats
df.describe(include='all')

# Check churn distribution
print(df['Churn'].value_counts())
sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.show()

# Gender vs Churn
gender_churn = pd.crosstab(df['gender'], df['Churn'], normalize='index')
print(gender_churn)
gender_churn.plot(kind='bar', stacked=True)
plt.title("Churn Rate by Gender")
plt.ylabel("Proportion")
plt.show()

# SeniorCitizen vs Churn
senior_churn = pd.crosstab(df['SeniorCitizen'], df['Churn'], normalize='index')
print(senior_churn)
senior_churn.plot(kind='bar', stacked=True)
plt.title("Churn Rate by Senior Citizen Status")
plt.ylabel("Proportion")
plt.show()

# Partner vs Churn
partner_churn = pd.crosstab(df['Partner'], df['Churn'], normalize='index')
print(partner_churn)
partner_churn.plot(kind='bar', stacked=True)
plt.title("Churn Rate by Partner Status")
plt.ylabel("Proportion")
plt.show()

sns.histplot(data=df, x='tenure', hue='Churn', multiple='stack', bins=30)
plt.title("Tenure vs Churn")
plt.show()

contract_churn = pd.crosstab(df['Contract'], df['Churn'], normalize='index')
print(contract_churn)
contract_churn.plot(kind='bar', stacked=True)
plt.title("Churn Rate by Contract Type")
plt.ylabel("Proportion")
plt.show()

payment_churn = pd.crosstab(df['PaymentMethod'], df['Churn'], normalize='index')
sns.heatmap(payment_churn, annot=True, fmt=".2f", cmap="Blues")
plt.title("Churn Rate by Payment Method")
plt.ylabel("Payment Method")
plt.xlabel("Churn")
plt.show()

# Convert churn to numeric (Yes=1, No=0)
df['Churn_Encoded'] = df['Churn'].apply(lambda x: 1 if x == "Yes" else 0)

# Focus on numeric columns
numeric_cols = df.select_dtypes(include=np.number)
corr = numeric_cols.corr()

sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# Make a copy to avoid altering original
df_ml = df.copy()

# Drop ID columns if present
if 'customerID' in df_ml.columns:
    df_ml.drop('customerID', axis=1, inplace=True)

# Encode all categorical columns using LabelEncoder
label_encoders = {}
for col in df_ml.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df_ml[col] = le.fit_transform(df_ml[col])
    label_encoders[col] = le

df_ml.head()

X = df_ml.drop('Churn', axis=1)   # predictors
y = df_ml['Churn']                # target variable (0/1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1]

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

roc_auc = roc_auc_score(y_test, y_pred_prob)
print("ROC-AUC Score:", roc_auc)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0,1], [0,1], 'k--')  # diagonal
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# Logistic regression coefficients
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

print(feature_importance)
explainer = shap.Explainer(model, X_train_scaled)

# Calculate SHAP values
shap_values = explainer(X_test_scaled)

# Summary plot (importance of features)
shap.summary_plot(shap_values, X_test, plot_type="bar")

#This shows which features drive churn predictions overall. Look at columns like gender, SeniorCitizen, Partner, Dependents — if they dominate the top, check if that’s justified.

shap.summary_plot(shap_values, X_test)

# This beeswarm plot shows direction of impact (positive = churn more likely, negative = churn less likely).
# Explain the first test sample
shap.plots.waterfall(shap_values[0])

from lime import lime_tabular

lime_explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_train_scaled,
    feature_names=X.columns.tolist(),
    class_names=['No Churn', 'Churn'],
    mode='classification'
)

# Explain one sample
idx = 0
exp = lime_explainer.explain_instance(
    X_test_scaled[idx],
    model.predict_proba,
    num_features=10
)
exp.show_in_notebook(show_table=True)