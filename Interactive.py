import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from imblearn.combine import SMOTEENN
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline

# Load dataset
st.title('Loan Default Prediction Dashboard')

uploaded_file = st.file_uploader('Upload your dataset (CSV format)', type=['csv'])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write('Data Preview:', df.head())

    # Split features/target
    X = df.drop(columns='Default')
    y = df['Default']
    
    # Identify column types
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=['object']).columns.tolist()

    # Transformers
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numerical_transformer = StandardScaler()

    # Preprocessor
    preprocessor = ColumnTransformer([
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

    # Model setup
    model = RandomForestClassifier(random_state=42)
    pipe = ImbPipeline([
        ('preprocess', preprocessor),
        ('clf', model)
    ])

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Model training
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"Model Accuracy: {acc:.4f}")
    st.text(classification_report(y_test, y_pred))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    st.pyplot(plt)


    # Feature importance
    if hasattr(pipe['clf'], 'feature_importances_'):
        # Extract numerical and categorical feature names
        categorical_names = pipe.named_steps['preprocess'].transformers_[1][1].fit(X[categorical_cols]).get_feature_names_out(categorical_cols)
        feature_names = numerical_cols + list(categorical_names)

        # Get feature importances
        feature_importances = pipe['clf'].feature_importances_
        feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
        feat_df = feat_df.sort_values(by='Importance', ascending=False).head(10)

        # Plot feature importances
        st.bar_chart(feat_df.set_index('Feature'))
