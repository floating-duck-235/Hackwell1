import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import warnings

warnings.filterwarnings("ignore")

# Machine Learning Libraries
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import joblib

# Visualization (optional)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns

    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False
    print("Matplotlib/Seaborn not available. Skipping visualizations.")


class ChronicCareRiskPredictor:
    """
    Complete ML pipeline for chronic care risk prediction
    """

    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_columns = None

    def generate_synthetic_data(self, n_patients=10000):
        """Generate synthetic healthcare dataset"""
        np.random.seed(self.random_state)
        random.seed(self.random_state)

        print(f"Generating synthetic dataset with {n_patients} patients...")

        # Generate patient IDs
        patient_ids = [f"PAT_{str(i).zfill(6)}" for i in range(1, n_patients + 1)]

        # Demographics
        ages = np.random.normal(65, 12, n_patients).astype(int)
        ages = np.clip(ages, 18, 95)

        sexes = np.random.choice(["Male", "Female"], n_patients, p=[0.48, 0.52])
        ethnicities = np.random.choice(
            ["White", "Black", "Hispanic", "Asian", "Other"],
            n_patients,
            p=[0.6, 0.15, 0.15, 0.08, 0.02],
        )

        # Physical measurements
        heights = np.random.normal(170, 10, n_patients)
        weights = np.random.normal(80, 15, n_patients)
        weights = np.clip(weights, 40, 200)
        bmis = weights / ((heights / 100) ** 2)

        # Vital signs
        systolic_bps = np.random.normal(130, 20, n_patients)
        diastolic_bps = np.random.normal(80, 12, n_patients)
        heart_rates = np.random.normal(72, 12, n_patients)
        respiratory_rates = np.random.normal(16, 3, n_patients)
        temperatures = np.random.normal(98.6, 0.8, n_patients)
        oxygen_saturations = np.random.normal(98, 2, n_patients)
        oxygen_saturations = np.clip(oxygen_saturations, 85, 100)

        # Lifestyle factors
        smoking_statuses = np.random.choice(
            ["Never", "Former", "Current"], n_patients, p=[0.5, 0.35, 0.15]
        )
        alcohol_uses = np.random.choice(
            ["None", "Moderate", "Heavy"], n_patients, p=[0.4, 0.5, 0.1]
        )
        exercise_frequencies = np.random.poisson(3, n_patients)
        diet_quality_scores = np.random.uniform(1, 10, n_patients)

        # Chronic conditions (with realistic correlations)
        diabetes_prob = 0.1 + 0.002 * (ages - 40) + 0.02 * np.maximum(0, bmis - 25)
        diabetes = np.random.binomial(1, np.clip(diabetes_prob, 0, 0.8), n_patients)

        hypertension_prob = 0.15 + 0.003 * (ages - 30) + 0.01 * np.maximum(0, bmis - 25)
        hypertension = np.random.binomial(
            1, np.clip(hypertension_prob, 0, 0.9), n_patients
        )

        heart_failure_prob = 0.05 + 0.002 * (ages - 50) + 0.1 * hypertension
        heart_failure = np.random.binomial(
            1, np.clip(heart_failure_prob, 0, 0.6), n_patients
        )

        ckd_prob = 0.08 + 0.002 * (ages - 45) + 0.05 * diabetes + 0.03 * hypertension
        chronic_kidney_disease = np.random.binomial(
            1, np.clip(ckd_prob, 0, 0.5), n_patients
        )

        obesity = (bmis >= 30).astype(int)

        # Lab values (influenced by conditions)
        hba1c_levels = np.random.normal(6.5, 1.5, n_patients) + 1.5 * diabetes
        hba1c_levels = np.clip(hba1c_levels, 4.0, 15.0)

        fasting_glucose = np.random.normal(100, 20, n_patients) + 50 * diabetes
        fasting_glucose = np.clip(fasting_glucose, 70, 400)

        total_cholesterol = np.random.normal(200, 40, n_patients)
        ldl_cholesterol = np.random.normal(120, 30, n_patients)
        hdl_cholesterol = np.random.normal(50, 15, n_patients)
        triglycerides = np.random.normal(150, 50, n_patients)

        # Kidney function
        serum_creatinine = (
            np.random.normal(1.0, 0.3, n_patients) + 0.5 * chronic_kidney_disease
        )
        serum_creatinine = np.clip(serum_creatinine, 0.5, 8.0)

        estimated_gfr = np.maximum(
            15, 120 - 30 * chronic_kidney_disease - 0.5 * (ages - 20)
        )

        # Additional features
        sodium_levels = np.random.normal(140, 5, n_patients)
        potassium_levels = np.random.normal(4.0, 0.5, n_patients)
        medication_adherence_rates = np.random.beta(8, 2, n_patients)
        medication_counts = (
            np.random.poisson(3, n_patients) + diabetes + hypertension + heart_failure
        )
        hospital_visits = (
            np.random.poisson(1, n_patients)
            + 2 * heart_failure
            + chronic_kidney_disease
        )
        er_visits = np.random.poisson(0.5, n_patients) + heart_failure
        days_since_last_visit = np.random.exponential(45, n_patients)

        # Mental health
        depression_prob = 0.15 + 0.05 * heart_failure + 0.03 * diabetes
        depression_diagnosis = np.random.binomial(
            1, np.clip(depression_prob, 0, 0.6), n_patients
        )

        anxiety_prob = 0.12 + 0.04 * heart_failure + 0.02 * diabetes
        anxiety_diagnosis = np.random.binomial(
            1, np.clip(anxiety_prob, 0, 0.5), n_patients
        )

        # Lifestyle scores
        sleep_quality_scores = np.random.uniform(1, 10, n_patients)
        systolic_bp_variations = np.random.exponential(10, n_patients)
        diastolic_bp_variations = np.random.exponential(8, n_patients)

        smoking_pack_years = np.where(
            smoking_statuses == "Never", 0, np.random.exponential(20, n_patients)
        )
        alcohol_units_per_week = np.where(
            alcohol_uses == "None",
            0,
            np.where(
                alcohol_uses == "Moderate",
                np.random.uniform(1, 14, n_patients),
                np.random.uniform(15, 50, n_patients),
            ),
        )

        physical_activity_levels = np.random.choice(
            ["Sedentary", "Low", "Moderate", "High"], n_patients, p=[0.3, 0.3, 0.3, 0.1]
        )

        # Outcome variable (realistic risk-based model)
        risk_score = (
            0.1 * (ages - 40) / 40
            + 0.2 * diabetes
            + 0.3 * heart_failure
            + 0.2 * chronic_kidney_disease
            + 0.1 * hypertension
            + 0.1 * (1 - medication_adherence_rates)
            + 0.05 * hospital_visits
            + 0.1 * depression_diagnosis
        )

        deterioration_prob = 1 / (1 + np.exp(-2 * (risk_score - 0.5)))
        outcome_90d_deterioration = np.random.binomial(
            1, deterioration_prob, n_patients
        )

        # Date fields
        base_date = datetime(2024, 1, 1)
        last_lab_dates = [
            base_date + timedelta(days=int(np.random.uniform(0, 365)))
            for _ in range(n_patients)
        ]
        last_visit_dates = [
            base_date + timedelta(days=int(np.random.uniform(0, 180)))
            for _ in range(n_patients)
        ]

        # Create DataFrame
        df = pd.DataFrame(
            {
                "patient_id": patient_ids,
                "age": ages,
                "sex": sexes,
                "ethnicity": ethnicities,
                "height_cm": np.round(heights, 1),
                "weight_kg": np.round(weights, 1),
                "BMI": np.round(bmis, 1),
                "systolic_BP": np.round(systolic_bps, 0).astype(int),
                "diastolic_BP": np.round(diastolic_bps, 0).astype(int),
                "heart_rate": np.round(heart_rates, 0).astype(int),
                "respiratory_rate": np.round(respiratory_rates, 0).astype(int),
                "temperature_F": np.round(temperatures, 1),
                "oxygen_saturation": np.round(oxygen_saturations, 1),
                "smoking_status": smoking_statuses,
                "alcohol_use": alcohol_uses,
                "exercise_frequency_per_week": exercise_frequencies,
                "diet_quality_score": np.round(diet_quality_scores, 1),
                "diabetes": diabetes,
                "hypertension": hypertension,
                "heart_failure": heart_failure,
                "chronic_kidney_disease": chronic_kidney_disease,
                "obesity": obesity,
                "hba1c_level": np.round(hba1c_levels, 1),
                "fasting_glucose_mg_dl": np.round(fasting_glucose, 0).astype(int),
                "total_cholesterol_mg_dl": np.round(total_cholesterol, 0).astype(int),
                "LDL_cholesterol_mg_dl": np.round(ldl_cholesterol, 0).astype(int),
                "HDL_cholesterol_mg_dl": np.round(hdl_cholesterol, 0).astype(int),
                "triglycerides_mg_dl": np.round(triglycerides, 0).astype(int),
                "serum_creatinine_mg_dl": np.round(serum_creatinine, 2),
                "estimated_GFR": np.round(estimated_gfr, 0).astype(int),
                "sodium_mmol_l": np.round(sodium_levels, 1),
                "potassium_mmol_l": np.round(potassium_levels, 1),
                "medication_adherence_rate": np.round(medication_adherence_rates, 2),
                "medication_count": medication_counts,
                "number_of_hospital_visits": hospital_visits,
                "number_of_er_visits": er_visits,
                "days_since_last_visit": np.round(days_since_last_visit, 0).astype(int),
                "depression_diagnosis": depression_diagnosis,
                "anxiety_diagnosis": anxiety_diagnosis,
                "sleep_quality_score": np.round(sleep_quality_scores, 1),
                "systolic_BP_variation": np.round(systolic_bp_variations, 1),
                "diastolic_BP_variation": np.round(diastolic_bp_variations, 1),
                "smoking_pack_years": np.round(smoking_pack_years, 1),
                "alcohol_units_per_week": np.round(alcohol_units_per_week, 1),
                "physical_activity_level": physical_activity_levels,
                "outcome_90d_deterioration": outcome_90d_deterioration,
                "last_lab_date": [date.strftime("%Y-%m-%d") for date in last_lab_dates],
                "last_clinical_visit_date": [
                    date.strftime("%Y-%m-%d") for date in last_visit_dates
                ],
            }
        )

        print(f"Dataset generated successfully! Shape: {df.shape}")
        return df

    def preprocess_data(self, df, fit_encoders=True):
        """Preprocess the dataset for machine learning"""
        data = df.copy()

        # Handle categorical variables
        categorical_cols = [
            "sex",
            "ethnicity",
            "smoking_status",
            "alcohol_use",
            "physical_activity_level",
        ]

        for col in categorical_cols:
            if fit_encoders:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
                self.label_encoders[col] = le
            else:
                if col in self.label_encoders:
                    data[col] = self.label_encoders[col].transform(data[col])

        # Convert dates to numeric
        reference_date = pd.to_datetime("2024-01-01")

        for date_col in ["last_lab_date", "last_clinical_visit_date"]:
            if date_col in data.columns:
                data[date_col] = pd.to_datetime(data[date_col])
                data[f"{date_col}_days"] = (data[date_col] - reference_date).dt.days
                data.drop(date_col, axis=1, inplace=True)

        # Remove patient_id
        if "patient_id" in data.columns:
            data.drop("patient_id", axis=1, inplace=True)

        # Separate features and target
        if "outcome_90d_deterioration" in data.columns:
            X = data.drop("outcome_90d_deterioration", axis=1)
            y = data["outcome_90d_deterioration"]
            if fit_encoders:
                self.feature_columns = X.columns.tolist()
            return X, y
        else:
            return data, None

    def train_models(self, X_train, y_train, quick_mode=False):
        """Train multiple models with hyperparameter tuning"""
        print("Training machine learning models...")

        if quick_mode:
            # Quick training without extensive hyperparameter tuning
            self.models = {
                "Random Forest": RandomForestClassifier(
                    n_estimators=100, random_state=self.random_state, n_jobs=-1
                ),
                "Gradient Boosting": GradientBoostingClassifier(
                    n_estimators=100, random_state=self.random_state
                ),
                "Logistic Regression": LogisticRegression(
                    random_state=self.random_state, max_iter=1000
                ),
            }

            # Train models
            for name, model in self.models.items():
                print(f"Training {name}...")
                if name == "Logistic Regression":
                    X_train_scaled = self.scaler.fit_transform(X_train)
                    model.fit(X_train_scaled, y_train)
                else:
                    model.fit(X_train, y_train)
        else:
            # Full hyperparameter tuning
            param_grids = {
                "Random Forest": {
                    "n_estimators": [100, 200],
                    "max_depth": [10, 20, None],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2],
                },
                "Gradient Boosting": {
                    "n_estimators": [100, 200],
                    "learning_rate": [0.05, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                },
                "Logistic Regression": {
                    "C": [0.1, 1.0, 10.0],
                    "penalty": ["l1", "l2"],
                    "solver": ["liblinear"],
                },
            }

            base_models = {
                "Random Forest": RandomForestClassifier(
                    random_state=self.random_state, n_jobs=-1
                ),
                "Gradient Boosting": GradientBoostingClassifier(
                    random_state=self.random_state
                ),
                "Logistic Regression": LogisticRegression(
                    random_state=self.random_state, max_iter=1000
                ),
            }

            self.models = {}

            for name, base_model in base_models.items():
                print(f"Tuning {name}...")

                if name == "Logistic Regression":
                    X_train_use = self.scaler.fit_transform(X_train)
                else:
                    X_train_use = X_train

                grid_search = GridSearchCV(
                    base_model,
                    param_grids[name],
                    cv=3,
                    scoring="roc_auc",
                    n_jobs=-1,
                    verbose=0,
                )

                grid_search.fit(X_train_use, y_train)
                self.models[name] = grid_search.best_estimator_
                print(f"Best {name} parameters: {grid_search.best_params_}")

    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        print("\nEvaluating models...")
        results = {}

        for name, model in self.models.items():
            if name == "Logistic Regression":
                X_test_use = self.scaler.transform(X_test)
            else:
                X_test_use = X_test

            # Predictions
            y_pred = model.predict(X_test_use)
            y_pred_proba = model.predict_proba(X_test_use)[:, 1]

            # Metrics
            results[name] = {
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "auc": roc_auc_score(y_test, y_pred_proba),
            }

            print(f"\n{name} Results:")
            for metric, value in results[name].items():
                print(f"  {metric.capitalize()}: {value:.4f}")

        # Find best model
        best_auc = max(results[name]["auc"] for name in results)
        self.best_model_name = [
            name for name in results if results[name]["auc"] == best_auc
        ][0]
        self.best_model = self.models[self.best_model_name]

        print(f"\nBest Model: {self.best_model_name} (AUC = {best_auc:.4f})")
        return results

    def get_feature_importance(self, top_n=15):
        """Get feature importance from the best model"""
        if self.best_model_name in ["Random Forest", "Gradient Boosting"]:
            feature_importance = pd.DataFrame(
                {
                    "feature": self.feature_columns,
                    "importance": self.best_model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)

            return feature_importance.head(top_n)
        else:
            print("Feature importance not available for this model type")
            return None

    def predict(self, X_new):
        """Make predictions on new data"""
        if self.best_model is None:
            raise ValueError("No model trained yet. Please train models first.")

        # Preprocess new data
        X_processed, _ = self.preprocess_data(X_new, fit_encoders=False)

        # Make predictions
        if self.best_model_name == "Logistic Regression":
            X_processed = self.scaler.transform(X_processed)

        probabilities = self.best_model.predict_proba(X_processed)[:, 1]
        predictions = self.best_model.predict(X_processed)

        return predictions, probabilities

    def save_model(self, filepath):
        """Save the trained model and preprocessors"""
        model_data = {
            "best_model": self.best_model,
            "best_model_name": self.best_model_name,
            "scaler": self.scaler,
            "label_encoders": self.label_encoders,
            "feature_columns": self.feature_columns,
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath):
        """Load a trained model and preprocessors"""
        model_data = joblib.load(filepath)
        self.best_model = model_data["best_model"]
        self.best_model_name = model_data["best_model_name"]
        self.scaler = model_data["scaler"]
        self.label_encoders = model_data["label_encoders"]
        self.feature_columns = model_data["feature_columns"]
        print(f"Model loaded from {filepath}")


def main():
    """Main execution function"""
    print("=" * 70)
    print("CHRONIC CARE RISK PREDICTION MODEL TRAINING")
    print("=" * 70)

    # Initialize the predictor
    predictor = ChronicCareRiskPredictor(random_state=42)

    # Generate synthetic data
    df = predictor.generate_synthetic_data(n_patients=10000)

    # Preprocess data
    X, y = predictor.preprocess_data(df)
    print(f"Dataset preprocessed: {X.shape[0]} samples, {X.shape[1]} features")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Train models (use quick_mode=True for faster training)
    predictor.train_models(X_train, y_train, quick_mode=True)

    # Evaluate models
    results = predictor.evaluate_models(X_test, y_test)

    # Show feature importance
    feature_importance = predictor.get_feature_importance()
    if feature_importance is not None:
        print(f"\nTop 15 Most Important Features ({predictor.best_model_name}):")
        print(feature_importance.to_string(index=False))

    # Save the model
    predictor.save_model("chronic_care_risk_model.pkl")

    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETED!")
    print("=" * 70)
    print(f"Best Model: {predictor.best_model_name}")
    print(f"Model saved as: chronic_care_risk_model.pkl")
    print("Use predictor.predict(new_data) to make predictions on new patients")

    return predictor


if __name__ == "__main__":
    predictor = main()
