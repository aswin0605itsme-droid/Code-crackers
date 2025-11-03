# Vital.AI: Intelligent Clinical Decision Support

[cite_start]This project is a prototype for **Vital.AI** [cite: 1][cite_start], an intelligent clinical decision support tool built for the HealthTech Hackathon[cite: 4].

[cite_start]Its goal is to transform complex patient data into rapid, actionable diagnostic insights, empowering medical professionals to make faster, more accurate clinical decisions[cite: 2].

## 🚀 The Core Engine
[cite_start]As outlined in our presentation, the core of this prototype is an AI Clinical Co-Pilot[cite: 14].

* [cite_start]**Language:** Python [cite: 45]
* [cite_start]**Web Framework:** Streamlit  (for the interactive UI)
* [cite_start]**Data/ML:** Pandas, Scikit-learn [cite: 47]
* [cite_start]**Model:** Random Forest Classifier 

## 🎯 Key Features
* [cite_start]**AI-Driven Analysis:** Predicts disease onset based on patient biometrics[cite: 17].
* [cite_start]**Explainable AI (XAI):** We don't just provide a "black box" prediction[cite: 67]. [cite_start]The app shows the key factors (e.g., Glucose, BMI, Age)  [cite_start]that contributed to the AI's diagnosis, building trust and enabling clinical verification[cite: 71].

## 🏃 How to Run This Project

1.  **Clone the repository:**
    ```bash
    git clone [your-github-repo-url]
    cd Vital.AI-project
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    
3.  **(One-Time Step) Train the model:**
    Run the training script to build the model file (`model.pkl`).
    ```bash
    python model_training.py
    ```

4.  **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```
    
5.  Open your browser and go to `http://localhost:8501` to use Vital.AI.