# 🩺 Patient Health Risk Prediction Model  

A machine learning project that predicts patient health risks based on medical records such as demographics, vitals, lifestyle factors, and lab results.  
This project demonstrates how AI can support early detection of health conditions and assist healthcare professionals in clinical decision-making.  

---

## 🚀 Features  
- 📊 Predicts risks for **diabetes, hypertension, heart failure, and kidney disease**  
- 🧠 Uses **LightGBM / XGBoost** for high-performance classification  
- ⚡ Trained on structured patient datasets with medical indicators  
- 🌐 Frontend UI (React/Node.js) for patient input and live predictions  
- 📈 Visualization of patient risk profile & health parameters  

---

## 📸 Project Snapshots  

👉 Add your frontend images/screenshots here:  

- **Patient Input Form (Frontend UI)**  
  ![Frontend Input Screenshot](images/frontend_input.png)  

- **Prediction Result Page**  
  ![Prediction Output Screenshot](images/frontend_output.png)  

- **Risk Visualization Dashboard**  
  ![Dashboard Screenshot](images/dashboard.png)  

---

## 🛠️ Tech Stack  

- **Frontend:** React.js, Tailwind CSS  
- **Backend:** Python (FastAPI / Flask)  
- **ML Models:** LightGBM, XGBoost, scikit-learn  
- **Deployment:** IBM Watsonx.ai, Watson Assistant (future integration)  

---

## 📂 Project Structure  

```
├── data/                  # Patient dataset (synthetic or anonymized)  
├── models/                # Trained ML models  
├── backend/               # API for model inference  
├── frontend/              # React UI  
├── notebooks/             # Training & evaluation notebooks  
└── README.md              # Project documentation  
```

---

## 📊 Example Prediction Flow  

1. User enters **patient details** (age, BMI, blood pressure, etc.)  
2. Model processes the input through encoders & ML pipeline  
3. Risk prediction is generated with probability scores  
4. Output is displayed in the **frontend dashboard**  

---

## 📌 Future Work  
- 🔗 Integration with **real-time EHR systems**  
- 🧾 Explainable AI for transparent predictions  
- ☁️ Deployment on **IBM Cloud / Dockerized service**  

---

## 🤝 Contributing  
Want to improve the project? Fork it, create a branch, and submit a PR 🚀  


