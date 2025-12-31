import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import time
import requests
import io
import base64
import os
import gdown

# ------------------ Constants ------------------ #
IMG_HEIGHT = 227
IMG_WIDTH = 227
CLASS_NAMES = ['Fresh', 'Rotten']
LANGUAGES = {"English": "en", "हिंदी": "hi", "తెలుగు": "te"}

# ------------------ Session State Initialization ------------------ #
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'user_points' not in st.session_state:
    st.session_state.user_points = 100
if 'carbon_saved' not in st.session_state:
    st.session_state.carbon_saved = 0.0
if 'food_waste_saved' not in st.session_state:
    st.session_state.food_waste_saved = 0.0
if 'achievements' not in st.session_state:
    st.session_state.achievements = []
if 'batch_results' not in st.session_state:
    st.session_state.batch_results = []

# Load model once
# ------------------ Load model once ------------------ #
import os
import gdown
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

@st.cache_resource
def load_alexnet_model():
    MODEL_PATH = "AlexNet_final.keras"
    GDRIVE_ID = "1Wq9yuL36YU1WBbqk_7_O6il9kbbdaXY0"

    GDRIVE_URL = f"https://drive.google.com/uc?id={GDRIVE_ID}"

    if not os.path.exists(MODEL_PATH):
        st.info("Downloading AlexNet model from Google Drive...")
        gdown.download(GDRIVE_URL, MODEL_PATH, quiet=False, fuzzy=True)

    try:
       model = load_model("AlexNet_final.keras")
        st.success("✅ AlexNet model loaded successfully")
        return model

    except Exception as e:
        st.error(f"❌ Model load failed: {e}")

        class MockModel:
            def predict(self, x):
                return np.random.uniform(0.3, 0.98, (x.shape[0], 2))

        return MockModel()

# ---- Call the function ---- #
model = load_model("AlexNet_final.keras")


# Preprocess image - FIXED for 227x227
def preprocess_image(image):
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = img_to_array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

# Enhanced preprocessing with image quality analysis
def enhanced_preprocess_image(image):
    """Advanced preprocessing with quality enhancement"""
    # Resize
    image = image.resize((IMG_WIDTH, IMG_HEIGHT))
    
    # Convert to array and normalize
    img_array = img_to_array(image) / 255.0
    
    # Apply enhancements
    pil_image = Image.fromarray((img_array * 255).astype(np.uint8))
    
    # Auto-contrast
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.2)
    
    # Sharpness
    enhancer = ImageEnhance.Sharpness(pil_image)
    pil_image = enhancer.enhance(1.1)
    
    enhanced_array = img_to_array(pil_image) / 255.0
    return np.expand_dims(enhanced_array, axis=0)

def analyze_image_quality(image):
    """Analyze image quality and provide feedback"""
    img_array = np.array(image)
    brightness = np.mean(img_array)
    contrast = np.std(img_array)
    quality_score = min(100, (brightness / 255) * 40 + (contrast / 128) * 60)
    
    feedback = []
    if brightness < 50:
        feedback.append(" Image is too dark")
    elif brightness > 200:
        feedback.append(" Image is overexposed")
    if contrast < 30:
        feedback.append(" Low contrast detected")
    
    return quality_score, feedback

# Freshness estimation days
def estimate_days(predicted_class):
    return 5 if predicted_class == "Fresh" else 1
# Sustainability tracking
class SustainabilityTracker:
    def __init__(self):
        self.carbon_saved = 0.0
        self.food_waste_saved = 0.0
        self.money_saved = 0.0
    def add_prediction(self, prediction, confidence):
        if prediction == "Fresh" and confidence > 0.7:
            carbon_saved = 0.1  
            waste_saved = 0.1   
            money_saved = 5.0   
            self.carbon_saved += carbon_saved
            self.food_waste_saved += waste_saved
            self.money_saved += money_saved
            return carbon_saved, waste_saved, money_saved
        return 0, 0, 0
sustainability_tracker = SustainabilityTracker()
# Gamification system
class GamificationSystem:
    def __init__(self):
        self.achievements = [
            {"name": " First Detection", "points": 10, "unlocked": False},
            {"name": " Fresh Expert", "points": 50, "unlocked": False},
            {"name": " Eco Warrior", "points": 100, "unlocked": False},
            {"name": " Analytics Pro", "points": 200, "unlocked": False}
        ]
    def check_achievements(self, user_points, predictions_count):
        unlocked = []
        if predictions_count >= 1 and not self.achievements[0]['unlocked']:
            self.achievements[0]['unlocked'] = True
            unlocked.append(self.achievements[0]['name'])
        if user_points >= 50 and not self.achievements[1]['unlocked']:
            self.achievements[1]['unlocked'] = True
            unlocked.append(self.achievements[1]['name'])
        return unlocked
gamification = GamificationSystem()
# Batch processing
def process_batch_images(uploaded_files):
    results = []
    progress_bar = st.progress(0)
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            image = Image.open(uploaded_file).convert('RGB')
            img_array = preprocess_image(image)
            prediction = model.predict(img_array, verbose=0)
            predicted_class = CLASS_NAMES[np.argmax(prediction)]
            confidence = np.max(prediction)
            results.append({
                'filename': uploaded_file.name,
                'prediction': predicted_class,
                'confidence': confidence,
                'timestamp': datetime.now()
            })
        except Exception as e:
            results.append({
                'filename': uploaded_file.name,
                'prediction': 'Error',
                'confidence': 0.0,
                'error': str(e)
            })
        progress_bar.progress((i + 1) / len(uploaded_files))
    return results
# Analytics and visualization
def create_analytics_dashboard():
    if not st.session_state.prediction_history:
        return None
    df = pd.DataFrame(st.session_state.prediction_history)
    # Create analytics charts
    col1, col2 = st.columns(2) 
    with col1:
        # Prediction distribution
        if 'prediction' in df.columns:
            fig1 = px.pie(df, names='prediction', title=' Prediction Distribution')
            st.plotly_chart(fig1, use_container_width=True)  
    with col2:
        # Confidence over time
        if 'timestamp' in df.columns and 'confidence' in df.columns:
            fig2 = px.line(df, x='timestamp', y='confidence', 
                          title=' Confidence Trend Over Time')
            st.plotly_chart(fig2, use_container_width=True)
    return df
# Background food images (rotating)
def background_slideshow_css():
    return """
   <style>
    .stApp {
        background: #ff69b4;  /* Hot Pink background */
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    .feature-card {
        background: rgba(255, 105, 180, 0.95);  /* Semi-transparent Hot Pink */
        color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        border: 2px solid rgba(255, 182, 193, 0.08);  /* Light Pink border */
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.12);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin: 5px;
    }
</style>
    """
# ------------------ Multi-language text ------------------ #
TEXTS = {
    "en": {
        "welcome": " Welcome to B Food Freshness Detector – Food Freshness Detection ",
        "choose_lang": "Select your preferred language:",
        "detection": " Food Freshness Detection",
        "upload": " Upload Image",
        "camera": " Camera Capture",
        "prediction": " Predicted Class",
        "days": " Estimated freshness duration",
        "calc": " Freshness-based Cost Calculator",
        "feedback": " Feedback",
        "cost": "Enter cost of the items kg (₹)",
        "qty": "Enter number of items",
        "submit": "Submit Feedback",
        "thankyou": " Thank you for your valuable feedback!",
        "batch_processing": " Batch Processing",
        "sustainability": " Sustainability Dashboard",
        "analytics": " Analytics & Reports",
        "settings": " Settings",
        "new_features": " New Features"
    },
    "hi": {
        "welcome": " एफएफडी – भोजन ताजगी जांच में आपका स्वागत है ",
        "choose_lang": "अपनी पसंदीदा भाषा चुनें:",
        "detection": " भोजन की ताजगी जांच",
        "upload": " छवि अपलोड करें",
        "camera": " कैमरा से लें",
        "prediction": " अनुमानित वर्ग",
        "days": " अनुमानित ताजगी अवधि",
        "calc": " ताजगी आधारित लागत कैलकुलेटर",
        "feedback": " प्रतिक्रिया",
        "cost": "वस्तु लागत दर्ज करें kg (₹)",
        "qty": "वस्तुओं की संख्या दर्ज करें",
        "submit": "प्रतिक्रिया जमा करें",
        "thankyou": " आपकी मूल्यवान प्रतिक्रिया के लिए धन्यवाद!",
        "batch_processing": " बैच प्रसंस्करण",
        "sustainability": " स्थिरता डैशबोर्ड",
        "analytics": " विश्लेषण और रिपोर्ट",
        "settings": " सेटिंग्स",
        "new_features": " नई सुविधाएं"
    },
    "te": {
        "welcome": " ఎఫ్‌ఎఫ్‌డీ – ఆహార తాజాదన నిర్ధారణకు స్వాగతం ",
        "choose_lang":"మీకు ఇష్టమైన భాషను ఎంచుకోండి:",
        "detection": " ఆహార తాజాదన నిర్ధారణ",
        "upload": " చిత్రం ఎక్కించండి",
        "camera": " కెమెరా ద్వారా తీసుకోండి",
        "prediction": " అంచనా ఫలితం",
        "days": " అంచనా తాజాదన వ్యవధి",
        "calc": " తాజాదన ఆధారిత ఖర్చు లెక్కింపు",
        "feedback": " అభిప్రాయం",
        "cost": "వస్తువు ధర kg (₹)",
        "qty": "వస్తువుల సంఖ్య",
        "submit": "అభిప్రాయం పంపించండి",
        "thankyou": " మీ విలువైన అభిప్రాయానికి ధన్యవాదాలు!",
        "batch_processing": " బ్యాచ్ ప్రాసెసింగ్",
        "sustainability": " సుస్థిరత డాష్బోర్డ్",
        "analytics": " విశ్లేషణలు & నివేదికలు",
        "settings": " సెట్టింగ్స్",
        "new_features": " కొత్త లక్షణాలు"
    }
}
# ------------------ Enhanced Tabs ------------------ #
tabs = st.tabs([" Welcome", " Detection", " Batch Mode", " Sustainability", 
               "Analytics", " Calculator", " Feedback", " Settings"])
# ---------- Tab 1: Welcome ---------- #
with tabs[0]:
    st.markdown(background_slideshow_css(), unsafe_allow_html=True)
    st.markdown("<h1 style='text-align:center; color:yellow;'>✨ Welcome to FFD ✨</h1>", unsafe_allow_html=True)
    st.write(" Choose your language:")
    lang = st.selectbox("Language", list(LANGUAGES.keys()))
    if lang:
        st.session_state["lang"] = LANGUAGES[lang] 
    # New Features Showcase
    if "lang" in st.session_state:
        t = TEXTS[st.session_state["lang"]]
        st.markdown("---")
        st.header(" " + t["new_features"])      
        col1, col2 = st.columns(2)      
        with col1:
            st.markdown('<div class="feature-card">'
                       '<h3> Batch Processing</h3>'
                       '<p>Upload multiple images at once for bulk analysis</p>'
                       '</div>', unsafe_allow_html=True)
            st.markdown('<div class="feature-card">'
                       '<h3> Sustainability Tracking</h3>'
                       '<p>Track your environmental impact and carbon savings</p>'
                       '</div>', unsafe_allow_html=True) 
        with col2:
            st.markdown('<div class="feature-card">'
                       '<h3> Advanced Analytics</h3>'
                       '<p>Interactive charts and detailed reports</p>'
                       '</div>', unsafe_allow_html=True)
            st.markdown('<div class="feature-card">'
                       '<h3> Gamification</h3>'
                       '<p>Earn points and unlock achievements</p>'
                       '</div>', unsafe_allow_html=True)
# ---------- Tab 2: Enhanced Detection ---------- #
with tabs[1]:
    st.markdown(background_slideshow_css(), unsafe_allow_html=True)

    if "lang" not in st.session_state:
        st.warning("⚠ Please select a language in Welcome tab first.")
    else:
        t = TEXTS[st.session_state["lang"]]
        st.header(t["detection"])

        option = st.radio("Input Method", [t["upload"], t["camera"]])
        image = None

        if option == t["upload"]:
            uploaded_file = st.file_uploader(t["upload"], type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file).convert('RGB')

        elif option == t["camera"]:
            camera_photo = st.camera_input(t["camera"])
            if camera_photo is not None:
                image = Image.open(camera_photo).convert('RGB')

        if image is not None:
            # Image Quality Analysis
            quality_score, feedback = analyze_image_quality(image)
            st.info(f" Image Quality Score: {quality_score:.1f}/100")
            
            if feedback:
                for msg in feedback:
                    st.warning(msg)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.image(image, caption="Selected Image", use_container_width=True)
            
            with col2:
                if st.button(" Analyze Freshness", type="primary"):
                    with st.spinner(" AI is analyzing..."):
                        try:
                            start_time = time.time()
                            img_array = enhanced_preprocess_image(image)
                            prediction = model.predict(img_array, verbose=0)
                            processing_time = time.time() - start_time
                            
                            predicted_class = CLASS_NAMES[np.argmax(prediction)]
                            confidence = np.max(prediction)
                            days = estimate_days(predicted_class)

                            # Sustainability tracking
                            carbon_saved, waste_saved, money_saved = sustainability_tracker.add_prediction(
                                predicted_class, confidence
                            )
                            
                            # Update session state
                            st.session_state.carbon_saved += carbon_saved
                            st.session_state.food_waste_saved += waste_saved
                            st.session_state.user_points += 5
                            
                            # Store prediction history
                            prediction_data = {
                                'timestamp': datetime.now(),
                                'prediction': predicted_class,
                                'confidence': confidence,
                                'processing_time': processing_time,
                                'carbon_saved': carbon_saved
                            }
                            st.session_state.prediction_history.append(prediction_data)
                            
                            # Display results
                            if predicted_class == "Fresh":
                                st.success(f" **{t['prediction']}: {predicted_class}** (Confidence: {confidence:.2%})")
                                st.balloons()
                            else:
                                st.error(f" **{t['prediction']}: {predicted_class}** (Confidence: {confidence:.2%})")
                            
                            st.info(f" **{t['days']}: {days} days**")
                            st.metric(" Processing Time", f"{processing_time:.2f}s")
                            
                            if carbon_saved > 0:
                                st.info(f" Environmental Impact: Saved {carbon_saved:.3f} kg CO2")
                            
                            # Check achievements
                            unlocked = gamification.check_achievements(
                                st.session_state.user_points, 
                                len(st.session_state.prediction_history)
                            )
                            
                            for achievement in unlocked:
                                if achievement not in st.session_state.achievements:
                                    st.session_state.achievements.append(achievement)
                                    st.success(f" Achievement Unlocked: {achievement}!")

                            # Save result for calculator tab
                            st.session_state["last_prediction"] = predicted_class
                            st.session_state["last_days"] = days

                        except Exception as e:
                            st.error(f" Error during prediction: {e}")

# ---------- Tab 3: Batch Processing ---------- #
with tabs[2]:
    st.markdown(background_slideshow_css(), unsafe_allow_html=True)

    if "lang" not in st.session_state:
        st.warning("⚠ Please select a language in Welcome tab first.")
    else:
        t = TEXTS[st.session_state["lang"]]
        st.header(" " + t["batch_processing"])
        
        uploaded_files = st.file_uploader(
            "Choose multiple images", 
            type=["jpg", "jpeg", "png"], 
            accept_multiple_files=True
        )
        
        if uploaded_files:
            if st.button(" Process All Images", type="primary"):
                results = process_batch_images(uploaded_files)
                st.session_state.batch_results = results
                
                # Display results
                results_df = pd.DataFrame(results)
                st.dataframe(results_df)
                
                # Summary
                fresh_count = len(results_df[results_df['prediction'] == 'Fresh'])
                rotten_count = len(results_df[results_df['prediction'] == 'Rotten'])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(" Fresh Items", fresh_count)
                with col2:
                    st.metric(" Rotten Items", rotten_count)
                
                # Export option
                if st.button(" Export Results as CSV"):
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
# ---------- Tab 4: Sustainability Dashboard ---------- #
with tabs[3]:
    st.markdown(background_slideshow_css(), unsafe_allow_html=True)
    if "lang" not in st.session_state:
        st.warning("⚠ Please select a language in Welcome tab first.")
    else:
        t = TEXTS[st.session_state["lang"]]
        st.header(" " + t["sustainability"])
        # Sustainability Metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'<div class="metric-card">'
                       f'<h3> Carbon Saved</h3>'
                       f'<h2>{st.session_state.carbon_saved:.3f} kg</h2>'
                       f'</div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card">'
                       f'<h3> Waste Prevented</h3>'
                       f'<h2>{st.session_state.food_waste_saved:.3f} kg</h2>'
                       f'</div>', unsafe_allow_html=True)
        with col3:
            st.markdown(f'<div class="metric-card">'
                       f'<h3> User Points</h3>'
                       f'<h2>{st.session_state.user_points}</h2>'
                       f'</div>', unsafe_allow_html=True)
        with col4:
            st.markdown(f'<div class="metric-card">'
                       f'<h3> Total Analyses</h3>'
                       f'<h2>{len(st.session_state.prediction_history)}</h2>'
                       f'</div>', unsafe_allow_html=True)
        # Achievements
        if st.session_state.achievements:
            st.subheader(" Your Achievements")
            for achievement in st.session_state.achievements:
                st.success(f" {achievement}")
# ---------- Tab 5: Analytics ---------- #
with tabs[4]:
    st.markdown(background_slideshow_css(), unsafe_allow_html=True)
    if "lang" not in st.session_state:
        st.warning("⚠ Please select a language in Welcome tab first.")
    else:
        t = TEXTS[st.session_state["lang"]]
        st.header(" " + t["analytics"])
        if st.session_state.prediction_history:
            df = create_analytics_dashboard()  
            # Detailed statistics
            st.subheader(" Detailed Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                total_predictions = len(st.session_state.prediction_history)
                st.metric("Total Predictions", total_predictions)
            with col2:
                avg_confidence = np.mean([p['confidence'] for p in st.session_state.prediction_history])
                st.metric("Average Confidence", f"{avg_confidence:.2%}")
            with col3:
                fresh_count = len([p for p in st.session_state.prediction_history if p['prediction'] == 'Fresh'])
                st.metric("Fresh Items", fresh_count)
            # Export data
            if st.button(" Export All Data"):
                data = {
                    'predictions': st.session_state.prediction_history,
                    'sustainability': {
                        'carbon_saved': st.session_state.carbon_saved,
                        'food_waste_saved': st.session_state.food_waste_saved,
                        'user_points': st.session_state.user_points
                    }
                }
                json_str = json.dumps(data, indent=2, default=str)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name="food_detection_data.json",
                    mime="application/json"
                )
        else:
            st.info(" No prediction data yet. Start analyzing images to see analytics!")
# ---------- Tab 6: Calculator ---------- #
with tabs[5]:
    st.markdown(background_slideshow_css(), unsafe_allow_html=True)

    if "lang" not in st.session_state:
        st.warning("⚠ Please select a language in Welcome tab first.")
    else:
        t = TEXTS[st.session_state["lang"]]
        st.header(t["calc"])

        if "last_prediction" in st.session_state:
            freshness = st.session_state["last_prediction"]
            days = st.session_state["last_days"]

            st.write(f"Last Prediction: **{freshness}** | {t['days']}: **{days}**")

            cost = st.number_input(t["cost"], min_value=1, value=50)
            qty = st.number_input(t["qty"], min_value=1, value=1)

            if freshness == "Fresh":
                total_cost = cost * qty
                st.success(f" Estimated Cost = ₹{total_cost:.2f} (Fresh - Full Price)")
            else:
                total_cost = cost * qty * 0.5  # 50% discount if rotten
                st.warning(f" Estimated Cost = ₹{total_cost:.2f} (Rotten - 50% Discount)")
        else:
            st.warning("⚠ Please run a prediction in Detection tab first!")
# ---------- Tab 7: Feedback ---------- #
with tabs[6]:
    st.markdown(background_slideshow_css(), unsafe_allow_html=True)
    if "lang" not in st.session_state:
        st.warning("⚠ Please select a language in Welcome tab first.")
    else:
        t = TEXTS[st.session_state["lang"]]
        st.header(t["feedback"])
        feedback = st.text_area(t["feedback"])
        if st.button(t["submit"]):
            st.success(t["thankyou"])
# ---------- Tab 8: Settings ---------- #
with tabs[7]:
    st.markdown(background_slideshow_css(), unsafe_allow_html=True)
    if "lang" not in st.session_state:
        st.warning("⚠ Please select a language in Welcome tab first.")
    else:
        t = TEXTS[st.session_state["lang"]]
        st.header("⚙️ " + t["settings"])
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Data Management")
            if st.button(" Clear Prediction History"):
                st.session_state.prediction_history = []
                st.success("History cleared!")  
            if st.button(" Reset Sustainability Data"):
                st.session_state.carbon_saved = 0.0
                st.session_state.food_waste_saved = 0.0
                st.session_state.user_points = 100
                st.session_state.achievements = []
                st.success("Sustainability data reset!")
        with col2:
            st.subheader("System Information")
            st.info("**Model:** AlexNet Fine-tuned")
            st.info("**Input Size:** 227x227 pixels")
            st.info("**Classes:** Fresh, Rotten")
            st.info("**Version:** 2.0 Enhanced")
# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: white;'>"
    "### 🌟 **Enhanced Food Freshness Detection System** • "
    "Built with  using Streamlit & TensorFlow"
    "</div>",
    unsafe_allow_html=True

)















