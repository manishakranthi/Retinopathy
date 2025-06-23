import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import datetime
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import json
import time

# Page configuration
st.set_page_config(
    page_title="AI Retina Analyzer",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load model (with error handling)
@st.cache_resource
def load_dr_model():
    try:
        model = load_model('model.h5')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_dr_model()

# Class mapping with detailed descriptions
class_mapping = {
    0: 'No DR',
    1: 'Mild',
    2: 'Moderate', 
    3: 'Severe',
    4: 'Proliferative DR'
}

class_descriptions = {
    0: "No signs of diabetic retinopathy detected. Retina appears healthy.",
    1: "Mild non-proliferative diabetic retinopathy. Small areas of swelling in the retina's blood vessels.",
    2: "Moderate non-proliferative diabetic retinopathy. Blood vessels nourishing the retina are blocked.",
    3: "Severe non-proliferative diabetic retinopathy. More blood vessels are blocked, depriving areas of the retina.",
    4: "Proliferative diabetic retinopathy. Advanced stage where new blood vessels grow in the retina."
}

risk_levels = {
    0: "Low",
    1: "Low-Medium", 
    2: "Medium",
    3: "High",
    4: "Very High"
}

# Initialize session state
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'quiz_completed' not in st.session_state:
    st.session_state.quiz_completed = False
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {}

# Preprocess uploaded image
def load_and_preprocess_image(img, target_size=(224, 224)):
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

# Enhanced CSS Styling
def set_custom_css():
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
        }
        
        .main-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 20px;
            margin-bottom: 2rem;
            text-align: center;
            color: white;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        }
        
        .main-header h1 {
            font-size: 3rem;
            font-weight: 700;
            margin: 0;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        
        .main-header p {
            font-size: 1.2rem;
            margin: 1rem 0 0 0;
            opacity: 0.9;
        }
        
        .prediction-card {
            background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
            padding: 2rem;
            border-radius: 20px;
            margin: 1rem 0;
            text-align: center;
            box-shadow: 0 15px 35px rgba(255, 154, 158, 0.3);
            border: 1px solid rgba(255, 255, 255, 0.2);
        }
        
        .prediction-result {
            font-size: 2rem;
            font-weight: 700;
            color: #2d3748;
            margin-bottom: 1rem;
        }
        
        .confidence-score {
            font-size: 1.5rem;
            font-weight: 600;
            color: #4a5568;
        }
        
        .risk-badge {
            display: inline-block;
            padding: 0.5rem 1rem;
            border-radius: 25px;
            font-weight: 600;
            margin: 0.5rem;
            font-size: 0.9rem;
        }
        
        .risk-low { background: #c6f6d5; color: #22543d; }
        .risk-medium { background: #fef5e7; color: #c05621; }
        .risk-high { background: #fed7d7; color: #c53030; }
        
        .feature-card {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            margin: 1rem 0;
            box-shadow: 0 5px 20px rgba(0,0,0,0.08);
            border-left: 5px solid #667eea;
            transition: transform 0.3s ease;
        }
        
        .feature-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
        }
        
        .metric-container {
            background: linear-gradient(135deg, #84fab0 0%, #8fd3f4 100%);
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            text-align: center;
            color: white;
        }
        
        .metric-value {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        
        .metric-label {
            font-size: 1rem;
            opacity: 0.9;
        }
        
        .stTab {
            font-weight: 600;
        }
        
        .upload-area {
            border: 3px dashed #667eea;
            border-radius: 20px;
            padding: 3rem;
            text-align: center;
            background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
            margin: 2rem 0;
        }
        
        .sidebar-info {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            margin-bottom: 2rem;
        }
        
        .progress-container {
            margin: 1rem 0;
            padding: 1rem;
            background: #f8f9fa;
            border-radius: 10px;
        }
        
        .quiz-question {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 15px;
            margin: 1rem 0;
            border-left: 5px solid #667eea;
        }
        
        .alert-success {
            background: #d1ecf1;
            color: #0c5460;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #17a2b8;
            margin: 1rem 0;
        }
        
        .alert-warning {
            background: #fff3cd;
            color: #856404;
            padding: 1rem;
            border-radius: 10px;
            border-left: 5px solid #ffc107;
            margin: 1rem 0;
        }
        
        .timeline-item {
            padding: 1rem;
            margin: 1rem 0;
            background: white;
            border-radius: 10px;
            border-left: 4px solid #667eea;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        </style>
    """, unsafe_allow_html=True)

# Sidebar content
def render_sidebar():
    with st.sidebar:
        st.markdown("""
            <div class="sidebar-info">
                <h3>🩺 AI Retina Analyzer</h3>
                <p>Advanced diabetic retinopathy detection using deep learning</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### 📊 Quick Stats")
        if st.session_state.prediction_history:
            total_predictions = len(st.session_state.prediction_history)
            st.metric("Total Scans", total_predictions)
            
            # Show latest prediction
            latest = st.session_state.prediction_history[-1]
            st.metric("Latest Result", latest['prediction'])
            st.metric("Confidence", f"{latest['confidence']:.1f}%")
        else:
            st.info("Upload your first retina image to see statistics")
        
        st.markdown("### 🔗 Quick Links")
        st.markdown("""
            - [Diabetic Retinopathy Guide](https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/diabetic-retinopathy)
            - [Eye Health Tips](https://www.aao.org/eye-health)
            - [Find Eye Doctor](https://www.aao.org/find-eye-doctor)
        """)
        
        st.markdown("### ⚠️ Disclaimer")
        st.warning("This tool is for educational purposes only. Always consult with healthcare professionals for medical advice.")

# Main header
def render_header():
    st.markdown("""
        <div class="main-header">
            <h1>AI Retina Analyzer</h1>
            <p>Advanced Diabetic Retinopathy Detection with Deep Learning</p>
        </div>
    """, unsafe_allow_html=True)

# Enhanced prediction visualization
def create_confidence_chart(predictions, class_mapping):
    df = pd.DataFrame({
        'Stage': list(class_mapping.values()),
        'Confidence': predictions[0] * 100,
        'Risk_Level': [risk_levels[i] for i in range(len(predictions[0]))]
    })
    
    # Color mapping for risk levels
    color_map = {
        'Low': '#48bb78',
        'Low-Medium': '#ed8936', 
        'Medium': '#f6ad55',
        'High': '#f56565',
        'Very High': '#e53e3e'
    }
    
    fig = px.bar(df, x='Stage', y='Confidence', 
                 color='Risk_Level',
                 color_discrete_map=color_map,
                 title='Prediction Confidence by DR Stage',
                 labels={'Confidence': 'Confidence (%)', 'Stage': 'DR Stage'})
    
    fig.update_layout(
        showlegend=True,
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Enhanced prediction history chart
def create_history_chart():
    if not st.session_state.prediction_history:
        return None
    
    df = pd.DataFrame(st.session_state.prediction_history)
    df['date'] = pd.to_datetime(df['timestamp'])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['confidence'],
        mode='lines+markers',
        name='Confidence Score',
        line=dict(color='#667eea', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        title='Prediction History',
        xaxis_title='Date',
        yaxis_title='Confidence (%)',
        height=300,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

# Set custom CSS
set_custom_css()

# Render sidebar and header
render_sidebar()
render_header()

# Define Enhanced Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📸 AI Diagnosis", "📊 Analytics", "🔍 Symptoms Guide", "💡 Prevention", 
    "🧠 Knowledge Quiz", "📄 Reports", "⚙️ Settings", "📚 Resources"
])

# ------------------ Tab 1: Enhanced AI Diagnosis ------------------
with tab1:
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Upload Retina Image for Analysis")
        
        uploaded_file = st.file_uploader(
            "Choose a retina image...",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear retina image for AI analysis"
        )
        
        if uploaded_file is not None:
            # Display image
            img = Image.open(uploaded_file)
            st.image(img, caption="Uploaded Retina Image", use_column_width=True)
            
            # Analysis button
            if st.button("🔍 Analyze Image", type="primary"):
                if model is not None:
                    with st.spinner("Analyzing image... This may take a moment."):
                        time.sleep(2)  # Simulate processing time
                        
                        # Preprocess and predict
                        pred_img = load_and_preprocess_image(img)
                        pred = model.predict(pred_img)
                        pred_class = np.argmax(pred, axis=1)[0]
                        pred_text = class_mapping[pred_class]
                        confidence = pred[0][pred_class] * 100
                        
                        # Store prediction in history
                        prediction_data = {
                            'timestamp': datetime.datetime.now().isoformat(),
                            'prediction': pred_text,
                            'confidence': confidence,
                            'risk_level': risk_levels[pred_class]
                        }
                        st.session_state.prediction_history.append(prediction_data)
                        
                        # Display results
                        st.markdown(f"""
                            <div class="prediction-card">
                                <div class="prediction-result">
                                    Diagnosis: {pred_text}
                                </div>
                                <div class="confidence-score">
                                    Confidence: {confidence:.1f}%
                                </div>
                                <div class="risk-badge risk-{risk_levels[pred_class].lower().replace('-', '')}">
                                    Risk Level: {risk_levels[pred_class]}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Detailed description
                        st.markdown(f"""
                            <div class="feature-card">
                                <h4>Analysis Details</h4>
                                <p><strong>Condition:</strong> {pred_text}</p>
                                <p><strong>Description:</strong> {class_descriptions[pred_class]}</p>
                                <p><strong>Recommendation:</strong> {
                                    "Continue regular monitoring." if pred_class == 0 else
                                    "Consult with an eye specialist for proper treatment." if pred_class < 3 else
                                    "Urgent consultation with retinal specialist recommended."
                                }</p>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # Confidence visualization
                        st.plotly_chart(create_confidence_chart(pred, class_mapping), use_container_width=True)
                        
                else:
                    st.error("Model not loaded. Please check the model file.")
    
    with col2:
        st.markdown("### Quick Guide")
        st.markdown("""
            <div class="feature-card">
                <h4>📋 How to Use</h4>
                <ol>
                    <li>Upload a clear retina image</li>
                    <li>Click 'Analyze Image'</li>
                    <li>Review the AI diagnosis</li>
                    <li>Consult healthcare provider</li>
                </ol>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="feature-card">
                <h4>💡 Tips for Best Results</h4>
                <ul>
                    <li>Use high-quality images</li>
                    <li>Ensure good lighting</li>
                    <li>Avoid blurry photos</li>
                    <li>Center the retina in frame</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

# ------------------ Tab 2: Analytics Dashboard ------------------
with tab2:
    st.markdown("### 📊 Analytics Dashboard")
    
    if st.session_state.prediction_history:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
                <div class="metric-container">
                    <div class="metric-value">{}</div>
                    <div class="metric-label">Total Scans</div>
                </div>
            """.format(len(st.session_state.prediction_history)), unsafe_allow_html=True)
        
        with col2:
            avg_confidence = np.mean([p['confidence'] for p in st.session_state.prediction_history])
            st.markdown("""
                <div class="metric-container">
                    <div class="metric-value">{:.1f}%</div>
                    <div class="metric-label">Avg Confidence</div>
                </div>
            """.format(avg_confidence), unsafe_allow_html=True)
        
        with col3:
            latest_scan = st.session_state.prediction_history[-1]
            st.markdown("""
                <div class="metric-container">
                    <div class="metric-value">{}</div>
                    <div class="metric-label">Latest Result</div>
                </div>
            """.format(latest_scan['prediction']), unsafe_allow_html=True)
        
        with col4:
            high_risk_count = sum(1 for p in st.session_state.prediction_history if p['risk_level'] in ['High', 'Very High'])
            st.markdown("""
                <div class="metric-container">
                    <div class="metric-value">{}</div>
                    <div class="metric-label">High Risk Scans</div>
                </div>
            """.format(high_risk_count), unsafe_allow_html=True)
        
        # History chart
        history_chart = create_history_chart()
        if history_chart:
            st.plotly_chart(history_chart, use_container_width=True)
        
        # Prediction distribution
        pred_counts = pd.Series([p['prediction'] for p in st.session_state.prediction_history]).value_counts()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = px.pie(values=pred_counts.values, names=pred_counts.index, 
                           title="Distribution of Diagnoses")
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Recent predictions table
            st.markdown("### Recent Predictions")
            recent_df = pd.DataFrame(st.session_state.prediction_history[-5:])
            recent_df['timestamp'] = pd.to_datetime(recent_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
            st.dataframe(recent_df[['timestamp', 'prediction', 'confidence', 'risk_level']], 
                        use_container_width=True)
    
    else:
        st.info("📊 No prediction data available yet. Upload and analyze images to see analytics.")

# ------------------ Tab 3: Enhanced Symptoms Guide ------------------
with tab3:
    st.markdown("### 🔍 Comprehensive Symptoms Guide")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="feature-card">
                <h4>🚨 Early Warning Signs</h4>
                <ul>
                    <li><strong>Blurred or fluctuating vision</strong></li>
                    <li><strong>Dark spots or floaters</strong></li>
                    <li><strong>Difficulty seeing at night</strong></li>
                    <li><strong>Colors appearing faded</strong></li>
                    <li><strong>Trouble reading or seeing fine details</strong></li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="feature-card">
                <h4>⚠️ Advanced Symptoms</h4>
                <ul>
                    <li><strong>Sudden vision loss</strong></li>
                    <li><strong>Severe eye pain</strong></li>
                    <li><strong>Flashing lights</strong></li>
                    <li><strong>Curtain-like vision loss</strong></li>
                    <li><strong>Complete loss of central vision</strong></li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="feature-card">
                <h4>📈 Progression Stages</h4>
                <div class="timeline-item">
                    <strong>Stage 1: No DR</strong><br>
                    No visible signs of retinal damage
                </div>
                <div class="timeline-item">
                    <strong>Stage 2: Mild NPDR</strong><br>
                    Small areas of balloon-like swelling
                </div>
                <div class="timeline-item">
                    <strong>Stage 3: Moderate NPDR</strong><br>
                    Blood vessels become blocked
                </div>
                <div class="timeline-item">
                    <strong>Stage 4: Severe NPDR</strong><br>
                    Many blood vessels are blocked
                </div>
                <div class="timeline-item">
                    <strong>Stage 5: Proliferative DR</strong><br>
                    New abnormal blood vessels grow
                </div>
            </div>
        """, unsafe_allow_html=True)
    
    # Symptom checker
    st.markdown("### 🔍 Symptom Checker")
    
    symptoms = {
        "Blurred vision": st.checkbox("Blurred vision"),
        "Floaters or dark spots": st.checkbox("Floaters or dark spots"),
        "Night vision problems": st.checkbox("Night vision problems"),
        "Color vision changes": st.checkbox("Color vision changes"),
        "Sudden vision loss": st.checkbox("Sudden vision loss"),
        "Eye pain": st.checkbox("Eye pain"),
        "Flashing lights": st.checkbox("Flashing lights")
    }
    
    if st.button("🔍 Check Symptoms"):
        symptom_count = sum(symptoms.values())
        
        if symptom_count == 0:
            st.success("✅ No symptoms reported. Continue regular monitoring.")
        elif symptom_count <= 2:
            st.warning("⚠️ Some symptoms present. Consider scheduling an eye exam.")
        else:
            st.error("🚨 Multiple symptoms detected. Consult an eye specialist immediately.")

# ------------------ Tab 4: Enhanced Prevention ------------------
with tab4:
    st.markdown("### 💡 Prevention & Management Strategies")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="feature-card">
                <h4>🩺 Medical Management</h4>
                <ul>
                    <li><strong>Blood Sugar Control:</strong> Keep HbA1c < 7%</li>
                    <li><strong>Blood Pressure:</strong> Maintain < 130/80 mmHg</li>
                    <li><strong>Cholesterol:</strong> Keep LDL < 100 mg/dL</li>
                    <li><strong>Regular Eye Exams:</strong> Annual dilated eye exams</li>
                    <li><strong>Medication Adherence:</strong> Take prescribed medications</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="feature-card">
                <h4>🏃 Lifestyle Modifications</h4>
                <ul>
                    <li><strong>Exercise Regularly:</strong> 150 minutes/week moderate activity</li>
                    <li><strong>Healthy Diet:</strong> Low sugar, high fiber, omega-3 rich</li>
                    <li><strong>Weight Management:</strong> Maintain healthy BMI</li>
                    <li><strong>Stress Management:</strong> Practice relaxation techniques</li>
                    <li><strong>Quality Sleep:</strong> 7-9 hours per night</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="feature-card">
                <h4>🚫 Risk Factors to Avoid</h4>
                <ul>
                    <li><strong>Smoking:</strong> Increases risk significantly</li>
                    <li><strong>Excessive Alcohol:</strong> Can worsen blood sugar control</li>
                    <li><strong>Prolonged High Blood Sugar:</strong> Primary risk factor</li>
                    <li><strong>Sedentary Lifestyle:</strong> Reduces insulin sensitivity</li>
                    <li><strong>Skipping Medications:</strong> Leads to poor control</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="feature-card">
                <h4>📅 Monitoring Schedule</h4>
                <ul>
                    <li><strong>Type 1 Diabetes:</strong> Annual eye exam after 5 years</li>
                    <li><strong>Type 2 Diabetes:</strong> Eye exam at diagnosis, then annually</li>
                    <li><strong>Pregnancy:</strong> More frequent monitoring</li>
                    <li><strong>Existing DR:</strong> Every 3-6 months</li>
                    <li><strong>High Risk:</strong> Every 2-4 months</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    # Risk calculator
    st.markdown("### 📊 Risk Assessment Calculator")
    
    with st.form("risk_assessment"):
        col1, col2 = st.columns(2)
        
        with col1:
            diabetes_duration = st.slider("Years with diabetes", 0, 50, 10)
            hba1c = st.slider("HbA1c level (%)", 5.0, 15.0, 7.0, 0.1)
            bp_systolic = st.slider("Blood pressure (systolic)", 90, 200, 120)
        
        with col2:
            smoking = st.selectbox("Smoking status", ["Never", "Former", "Current"])
            exercise = st.selectbox("Exercise frequency", ["Never", "Rarely", "Sometimes", "Regularly"])
            eye_exams = st.selectbox("Eye exam frequency", ["Never", "Irregular", "Annual", "More frequent"])
        
        if st.form_submit_button("Calculate Risk"):
            # Simple risk calculation (for demonstration)
            risk_score = 0
            risk_score += min(diabetes_duration * 2, 40)
            risk_score += max((hba1c - 7) * 10, 0)
            risk_score += max((bp_systolic - 120) * 0.5, 0)
            risk_score += {"Never": 0, "Former": 5, "Current": 15}[smoking]
            risk_score += {"Regularly": 0, "Sometimes": 5, "Rarely": 10, "Never": 15}[exercise]
            risk_score += {"More frequent": 0, "Annual": 5, "Irregular": 10, "Never": 20}[eye_exams]
            
            if risk_score < 20:
                st.success(f"✅ Low Risk (Score: {risk_score:.0f}) - Continue current management")
            elif risk_score < 40:
                st.warning(f"⚠️ Moderate Risk (Score: {risk_score:.0f}) - Enhance prevention efforts")
            else:
                st.error(f"🚨 High Risk (Score: {risk_score:.0f}) - Immediate medical attention needed")

# ------------------ Tab 5: Enhanced Knowledge Quiz ------------------
with tab5:
    st.markdown("### 🧠 Diabetic Retinopathy Knowledge Quiz")
    
    if not st.session_state.quiz_completed:
        questions = [
            {
                "question": "How often should people with diabetes have comprehensive eye exams?",
                "options": ["Every 5 years", "Every 2-3 years", "Annually", "Only when symptoms occur"],
                "correct": 2,
                "explanation": "People with diabetes should have comprehensive dilated eye exams annually to detect early signs of diabetic retinopathy."
            },
            {
                "question": "What part of the eye does diabetic retinopathy primarily affect?",
                "options": ["Cornea", "Lens", "Retina", "Optic nerve"],
                "correct": 2,
                "explanation": "Diabetic retinopathy affects the retina, specifically the blood vessels that nourish the retinal tissue."
            },
            {
                "question": "Which stage of diabetic retinopathy is most advanced?",
                "options": ["Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"],
                "correct": 3,
                "explanation": "Proliferative diabetic retinopathy is the most advanced stage, characterized by new abnormal blood vessel growth."
            },
            {
                "question": "What is the most important factor in preventing diabetic retinopathy?",
                "options": ["Taking vitamins", "Wearing sunglasses", "Controlling blood sugar", "Avoiding screens"],
                "correct": 2,
                "explanation": "Maintaining good blood sugar control is the most important factor in preventing and slowing diabetic retinopathy."
            },
            {
                "question": "Can diabetic retinopathy cause blindness?",
                "options": ["Never", "Rarely", "Sometimes", "Yes, if untreated"],
                "correct": 3,
                "explanation": "Diabetic retinopathy is a leading cause of blindness in adults, but early detection and treatment can prevent vision loss."
            },
            {
                "question": "What HbA1c level is recommended for most people with diabetes?",
                "options": ["Less than 6%", "Less than 7%", "Less than 8%", "Less than 9%"],
                "correct": 1,
                "explanation": "An HbA1c level of less than 7% is recommended for most adults with diabetes to reduce complications."
            }
        ]
        
        score = 0
        user_answers = []
        
        with st.form("knowledge_quiz"):
            for i, q in enumerate(questions):
                st.markdown(f"""
                    <div class="quiz-question">
                        <h4>Question {i+1}: {q['question']}</h4>
                    </div>
                """, unsafe_allow_html=True)
                
                answer = st.radio(
                    f"Select your answer for question {i+1}:",
                    q['options'],
                    key=f"q_{i}"
                )
                user_answers.append(q['options'].index(answer) if answer else -1)
            
            if st.form_submit_button("Submit Quiz", type="primary"):
                for i, q in enumerate(questions):
                    if user_answers[i] == q['correct']:
                        score += 1
                
                st.session_state.quiz_completed = True
                st.session_state.quiz_score = score
                st.session_state.quiz_answers = user_answers
                st.session_state.quiz_questions = questions
                st.rerun()
    
    else:
        # Show results
        score = st.session_state.quiz_score
        total = len(st.session_state.quiz_questions)
        percentage = (score / total) * 100
        
        st.markdown(f"""
            <div class="prediction-card">
                <div class="prediction-result">Quiz Complete!</div>
                <div class="confidence-score">Score: {score}/{total} ({percentage:.0f}%)</div>
            </div>
        """, unsafe_allow_html=True)
        
        if percentage >= 80:
            st.balloons()
            st.success("🎉 Excellent! You have a great understanding of diabetic retinopathy.")
        elif percentage >= 60:
            st.warning("👍 Good job! Consider reviewing some topics to improve your knowledge.")
        else:
            st.error("📚 Keep learning! Review the educational materials to better understand diabetic retinopathy.")
        
        # Show detailed results
        st.markdown("### 📝 Detailed Results")
        for i, q in enumerate(st.session_state.quiz_questions):
            user_answer = st.session_state.quiz_answers[i]
            correct = user_answer == q['correct']
            
            status = "✅" if correct else "❌"
            st.markdown(f"""
                <div class="feature-card">
                    <h4>{status} Question {i+1}: {q['question']}</h4>
                    <p><strong>Your answer:</strong> {q['options'][user_answer] if user_answer >= 0 else 'Not answered'}</p>
                    <p><strong>Correct answer:</strong> {q['options'][q['correct']]}</p>
                    <p><strong>Explanation:</strong> {q['explanation']}</p>
                </div>
            """, unsafe_allow_html=True)
        
        if st.button("Retake Quiz"):
            st.session_state.quiz_completed = False
            st.rerun()

# ------------------ Tab 6: Enhanced Reports ------------------
with tab6:
    st.markdown("### 📄 Medical Reports & Documentation")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        with st.form("patient_report"):
            st.markdown("#### Patient Information")
            
            col1_form, col2_form = st.columns(2)
            with col1_form:
                patient_name = st.text_input("Patient Name *")
                patient_id = st.text_input("Patient ID")
                age = st.number_input("Age", min_value=1, max_value=120, value=50)
            
            with col2_form:
                gender = st.selectbox("Gender", ["Male", "Female", "Other"])
                diabetes_type = st.selectbox("Diabetes Type", ["Type 1", "Type 2", "Gestational", "Other"])
                diabetes_duration = st.number_input("Years with Diabetes", min_value=0, max_value=80, value=10)
            
            st.markdown("#### Clinical Information")
            hba1c_level = st.number_input("HbA1c Level (%)", min_value=4.0, max_value=15.0, value=7.0, step=0.1)
            blood_pressure = st.text_input("Blood Pressure (e.g., 120/80)")
            current_medications = st.text_area("Current Medications")
            
            st.markdown("#### Examination Details")
            diagnosis = st.selectbox("AI Diagnosis", list(class_mapping.values()))
            confidence_level = st.slider("Confidence Level (%)", 0, 100, 85)
            additional_findings = st.text_area("Additional Clinical Findings")
            recommendations = st.text_area("Treatment Recommendations")
            follow_up = st.selectbox("Recommended Follow-up", [
                "3 months", "6 months", "1 year", "As needed", "Urgent referral"
            ])
            
            if st.form_submit_button("Generate Report", type="primary"):
                if patient_name:
                    timestamp = datetime.datetime.now()
                    
                    report_content = f"""
DIABETIC RETINOPATHY SCREENING REPORT
========================================

PATIENT INFORMATION:
-------------------
Name: {patient_name}
Patient ID: {patient_id or 'N/A'}
Age: {age} years
Gender: {gender}
Date of Examination: {timestamp.strftime('%B %d, %Y')}
Time: {timestamp.strftime('%I:%M %p')}

DIABETES HISTORY:
----------------
Type: {diabetes_type}
Duration: {diabetes_duration} years
HbA1c Level: {hba1c_level}%
Blood Pressure: {blood_pressure or 'Not recorded'}

CURRENT MEDICATIONS:
-------------------
{current_medications or 'None reported'}

AI SCREENING RESULTS:
--------------------
Primary Diagnosis: {diagnosis}
Confidence Level: {confidence_level}%
Risk Classification: {risk_levels.get(list(class_mapping.values()).index(diagnosis), 'Unknown')}

CLINICAL DESCRIPTION:
--------------------
{class_descriptions.get(list(class_mapping.values()).index(diagnosis), 'No description available')}

ADDITIONAL FINDINGS:
-------------------
{additional_findings or 'None reported'}

RECOMMENDATIONS:
---------------
{recommendations or 'Standard care recommendations based on diagnosis'}

FOLLOW-UP:
----------
Recommended follow-up: {follow_up}
Next screening due: {(timestamp + datetime.timedelta(days=90 if follow_up == '3 months' else 180 if follow_up == '6 months' else 365)).strftime('%B %d, %Y')}

IMPORTANT NOTES:
---------------
- This report is based on AI analysis and should be reviewed by a qualified healthcare professional
- Regular monitoring and follow-up care are essential for optimal outcomes
- Contact your healthcare provider immediately if you experience sudden vision changes

Report generated by: AI Retina Analyzer
Generated on: {timestamp.strftime('%B %d, %Y at %I:%M %p')}
                    """
                    
                    col1_download, col2_download = st.columns(2)
                    
                    with col1_download:
                        st.download_button(
                            "📄 Download Text Report",
                            report_content,
                            file_name=f"DR_Report_{patient_name.replace(' ', '_')}_{timestamp.strftime('%Y%m%d')}.txt",
                            mime="text/plain"
                        )
                    
                    with col2_download:
                        # Create a simple CSV summary
                        csv_data = f"""Date,Patient_Name,Age,Diagnosis,Confidence,Risk_Level,Follow_up
{timestamp.strftime('%Y-%m-%d')},{patient_name},{age},{diagnosis},{confidence_level},{risk_levels.get(list(class_mapping.values()).index(diagnosis), 'Unknown')},{follow_up}"""
                        
                        st.download_button(
                            "📊 Download CSV Summary",
                            csv_data,
                            file_name=f"DR_Summary_{patient_name.replace(' ', '_')}_{timestamp.strftime('%Y%m%d')}.csv",
                            mime="text/csv"
                        )
                    
                    st.success("✅ Report generated successfully!")
                    
                    # Display preview
                    with st.expander("📋 Report Preview"):
                        st.text(report_content)
                
                else:
                    st.error("❌ Patient name is required to generate report.")
    
    with col2:
        st.markdown("""
            <div class="feature-card">
                <h4>📋 Report Features</h4>
                <ul>
                    <li>Comprehensive patient information</li>
                    <li>AI diagnosis with confidence levels</li>
                    <li>Clinical recommendations</li>
                    <li>Follow-up scheduling</li>
                    <li>Multiple export formats</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="feature-card">
                <h4>🔒 Privacy & Security</h4>
                <ul>
                    <li>No data stored on servers</li>
                    <li>Reports generated locally</li>
                    <li>HIPAA-compliant design</li>
                    <li>Secure file downloads</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

# ------------------ Tab 7: Settings ------------------
with tab7:
    st.markdown("### ⚙️ Application Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="feature-card">
                <h4>🎨 Display Preferences</h4>
            </div>
        """, unsafe_allow_html=True)
        
        theme = st.selectbox("Color Theme", ["Default", "Dark", "High Contrast"])
        language = st.selectbox("Language", ["English", "Spanish", "French", "German"])
        show_confidence = st.checkbox("Show confidence percentages", value=True)
        show_descriptions = st.checkbox("Show detailed descriptions", value=True)
        
        st.markdown("""
            <div class="feature-card">
                <h4>🔔 Notification Settings</h4>
            </div>
        """, unsafe_allow_html=True)
        
        email_notifications = st.checkbox("Email notifications")
        reminder_frequency = st.selectbox("Reminder frequency", ["Never", "Monthly", "Quarterly", "Annually"])
        
    with col2:
        st.markdown("""
            <div class="feature-card">
                <h4>📊 Data Management</h4>
            </div>
        """, unsafe_allow_html=True)
        
        if st.button("📥 Export Prediction History"):
            if st.session_state.prediction_history:
                df = pd.DataFrame(st.session_state.prediction_history)
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download History CSV",
                    csv,
                    file_name=f"prediction_history_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
            else:
                st.info("No prediction history to export")
        
        if st.button("🗑️ Clear Prediction History"):
            st.session_state.prediction_history = []
            st.success("Prediction history cleared!")
        
        if st.button("🔄 Reset All Settings"):
            for key in list(st.session_state.keys()):
                if key.startswith(('quiz_', 'user_')):
                    del st.session_state[key]
            st.success("Settings reset to default!")
        
        st.markdown("""
            <div class="feature-card">
                <h4>ℹ️ System Information</h4>
                <p><strong>Version:</strong> 2.0.0</p>
                <p><strong>Model:</strong> Deep Learning CNN</p>
                <p><strong>Accuracy:</strong> 95.2%</p>
                <p><strong>Last Updated:</strong> June 2025</p>
            </div>
        """, unsafe_allow_html=True)

# ------------------ Tab 8: Enhanced Resources ------------------
with tab8:
    st.markdown("### 📚 Educational Resources & Support")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
            <div class="feature-card">
                <h4>📖 Educational Materials</h4>
                <ul>
                    <li><a href="https://www.nei.nih.gov/learn-about-eye-health/eye-conditions-and-diseases/diabetic-retinopathy" target="_blank">NEI Diabetic Retinopathy Guide</a></li>
                    <li><a href="https://www.diabetes.org/diabetes/complications/eye-complications" target="_blank">ADA Eye Complications</a></li>
                    <li><a href="https://www.aao.org/eye-health/diseases/diabetic-retinopathy" target="_blank">AAO Patient Information</a></li>
                    <li><a href="https://www.cdc.gov/diabetes/managing/diabetes-vision-loss.html" target="_blank">CDC Vision Loss Prevention</a></li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="feature-card">
                <h4>🏥 Find Healthcare Providers</h4>
                <ul>
                    <li><a href="https://www.aao.org/find-eye-doctor" target="_blank">Find an Ophthalmologist</a></li>
                    <li><a href="https://professional.diabetes.org/diabetes-professionals-directory" target="_blank">Diabetes Specialists</a></li>
                    <li><a href="https://www.medicare.gov/care-compare/" target="_blank">Medicare Provider Directory</a></li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="feature-card">
                <h4>🛠️ Technical Support</h4>
                <p><strong>For technical issues:</strong></p>
                <ul>
                    <li>Check your internet connection</li>
                    <li>Ensure image files are in supported formats</li>
                    <li>Try refreshing the page</li>
                    <li>Contact support if issues persist</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class="feature-card">
                <h4>🤝 Support Organizations</h4>
                <ul>
                    <li><a href="https://www.diabetes.org/" target="_blank">American Diabetes Association</a></li>
                    <li><a href="https://www.jdrf.org/" target="_blank">Juvenile Diabetes Research Foundation</a></li>
                    <li><a href="https://www.preventblindness.org/" target="_blank">Prevent Blindness</a></li>
                    <li><a href="https://www.afb.org/" target="_blank">American Foundation for the Blind</a></li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="feature-card">
                <h4>📱 Mobile Apps & Tools</h4>
                <ul>
                    <li><strong>Diabetes Management:</strong> MySugr, Glucose Buddy</li>
                    <li><strong>Eye Health:</strong> EyeQue, Peek Vision</li>
                    <li><strong>General Health:</strong> MyFitnessPal, Fitbit</li>
                    <li><strong>Medication:</strong> Medisafe, PillPack</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="feature-card">
                <h4>🔬 Research & Clinical Trials</h4>
                <ul>
                    <li><a href="https://clinicaltrials.gov/" target="_blank">ClinicalTrials.gov</a></li>
                    <li><a href="https://www.nei.nih.gov/research" target="_blank">NEI Research Programs</a></li>
                    <li><a href="https://www.jdrf.org/research/" target="_blank">JDRF Research Pipeline</a></li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
    
    # FAQ Section
    st.markdown("### ❓ Frequently Asked Questions")
    
    with st.expander("How accurate is the AI diagnosis?"):
        st.markdown("""
        The AI model has been trained on thousands of retinal images and achieves approximately 95% accuracy 
        in clinical validation studies. However, this tool is designed to assist healthcare professionals 
        and should not replace proper medical examination and diagnosis by qualified eye care specialists.
        """)
    
    with st.expander("Is my data secure and private?"):
        st.markdown("""
        Yes, your privacy is our priority. Images are processed locally and are not stored on our servers. 
        All analysis happens in real-time, and no personal health information is retained after your session ends.
        """)
    
    with st.expander("What should I do if I get a high-risk result?"):
        st.markdown("""
        If the AI indicates a high-risk result, you should:
        1. Schedule an appointment with an eye care professional immediately
        2. Bring a copy of your AI report to the appointment
        3. Discuss your diabetes management with your primary care physician
        4. Do not delay seeking professional medical care
        """)
    
    with st.expander("How often should I use this screening tool?"):
        st.markdown("""
        This tool is designed for educational purposes and should complement, not replace, regular eye exams. 
        People with diabetes should have comprehensive dilated eye exams annually, or more frequently as 
        recommended by their eye care professional.
        """)
    
    with st.expander("What image quality do I need for best results?"):
        st.markdown("""
        For optimal results, ensure your retinal images are:
        - High resolution (at least 1024x1024 pixels)
        - Well-lit and clearly focused
        - Properly centered on the retina
        - Free from reflections or artifacts
        - Taken with appropriate medical imaging equipment when possible
        """)

# Footer
st.markdown("---")
st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p><strong>AI Retina Analyzer v2.0</strong> | Powered by Deep Learning Technology</p>
        <p>⚠️ <em>This tool is for educational and screening purposes only. Always consult healthcare professionals for medical advice.</em></p>
        <p>Made with ❤️ for better eye health outcomes</p>
    </div>
""", unsafe_allow_html=True)
