🎯 Overview
This application uses a Support Vector Machine (SVM) classifier trained on real patient data to assess diabetes risk. Users input their health metrics, and the system provides:

Risk probability percentage
Confidence score
Validation status
Personalized health recommendations
Interactive visualizations

The model analyzes 12 key health indicators to make predictions with approximately 82-85% accuracy.

✨ Features
🔍 Health Assessment

Comprehensive 12-metric health evaluation
Real-time risk prediction
Probability-based scoring system
Confidence level indicators

📱 Progressive Web App (PWA)

Installable on mobile devices
Offline capability
Fullscreen mode (no URL bar)
Custom app icon
Native app experience

💡 Smart Features

Auto-calculated BMI
Real-time input validation
Personalized recommendations
Dataset exploration tools
Correlation heatmaps

🛠️ Technology Stack
Core Technologies

Python 3.9+ - Programming language
Streamlit 1.25+ - Web framework
scikit-learn 1.3+ - Machine learning
Pandas 2.0+ - Data manipulation
NumPy 1.23+ - Numerical computing
Matplotlib 3.7+ - Visualization
Seaborn 0.12+ - Statistical visualization

Deployment

Streamlit Cloud - Hosting platform
GitHub - Version control
PWA - Progressive Web App features


📥 Installation
Prerequisites

Python 3.9 or higher
pip package manager
Git (optional)

Option 1: Quick Start (Recommended)

Clone the repository:

bash   git clone https://github.com/intanabdali/Diabetes-Prediction-App.git
   cd Diabetes-Prediction-App

Create virtual environment:

bash   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate

Install dependencies:

bash   pip install -r requirements.txt

Run the application:

bash   streamlit run diabetes_prediction_web_app.py

Open in browser:

The app will automatically open at http://localhost:8501
If not, manually navigate to that URL



Option 2: Direct Installation
bash# Install dependencies
pip install streamlit pandas numpy scikit-learn matplotlib seaborn

# Run application
streamlit run diabetes_prediction_web_app.py

🚀 Usage
Making a Prediction

Navigate to Prediction Page

Open the app in your browser
Ensure "Prediction" is selected in the sidebar


Enter Health Information
Physical Statistics:

Gender (Male/Female)
Height (cm)
Weight (kg)
BMI (auto-calculated)

Vital Indicators:

Resting Heart Rate (BPM)
Glucose Level (mg/dL)
Systolic Blood Pressure (mmHg)
Diastolic Blood Pressure (mmHg)
Age (years)

Clinical History:

☐ Hypertension
☐ Cardiovascular Disease
☐ Stroke History


Click "Analyze Risk Factor"

View risk percentage
Check confidence score
Read validation status
Review recommendations

Installing as Mobile App
On Android:

Open app in Chrome browser
Tap menu (⋮) → "Install app" or "Add to Home screen"
Tap "Install"
App icon appears on home screen
Opens fullscreen like a native app

On iOS:

Open app in Safari browser
Tap Share button (⬆️)
Tap "Add to Home Screen"
Tap "Add"
App icon appears on home screen


🧠 Model Information
Algorithm

Type: Support Vector Machine (SVM)
Kernel: Linear
Training Data: 5,289 patient records
Features: 12 health indicators

Performance Metrics

Accuracy: ~82-85%
Sensitivity: ~65-70% (diabetic detection)
Specificity: ~85-90% (non-diabetic detection)

Feature Importance

Glucose (Most Important) - Direct diabetes indicator
BMI - Strong correlation with Type 2 diabetes
Age - Risk increases with age
Blood Pressure - Often co-occurs with diabetes
Hypertension - Linked to metabolic syndrome

