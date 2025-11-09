# 🧠 Day 13 — AI Resume Screener

### Project Overview
This project focuses on automating the **resume screening process** using **Natural Language Processing (NLP)**.  
By leveraging TF-IDF vectorization and a Naive Bayes classifier, the model predicts which category a candidate’s resume best fits (e.g., Data Science, AI, Design, etc.).

---

### 🔍 Objective
To build a simple yet effective machine learning pipeline that can:
- Extract and process textual data from resumes  
- Convert text into numerical form using **TF-IDF**  
- Classify resumes into categories with **Naive Bayes**  
- Evaluate and visualize results via a **confusion matrix**

---

### ⚙️ Technical Stack
Python | Scikit-learn | Pandas | NumPy | Matplotlib | Seaborn | TF-IDF | NLP  

---

### 🧩 Workflow
1. **Data Loading** — Load and inspect the resume dataset  
2. **Preprocessing** — Handle small sample classes, clean text  
3. **Feature Extraction** — TF-IDF vectorization for text  
4. **Model Training** — Multinomial Naive Bayes for classification  
5. **Evaluation** — Cross-validation and confusion matrix  
6. **Export** — Save trained pipeline and visualization outputs  

---

### 📊 Results
- **Cross-validated Accuracy:** 60% (demo dataset)  
- **Visualization:** `resume_confusion_matrix.png`  
- **Saved Model:** `resume_screening_model.pkl`  

> Note: The dataset used here is for demonstration. Real-world data with richer samples can yield higher performance.

---

### 🧠 Insights
Even with limited data, TF-IDF and Naive Bayes demonstrate the strength of **interpretable NLP** models for candidate classification — forming the foundation for AI-powered HR tools.

---

### ▶️ How to Run
```bash
source ../Day-01-Titanic/venv/bin/activate
python3 run_resume_screener.py