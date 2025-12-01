import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Custom CSS for enhanced UI/UX
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-green: #2d6a4f;
        --secondary-green: #52b788;
        --accent-gold: #f4a261;
        --bg-light: #f8f9fa;
        --text-dark: #212529;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Custom header styling */
    .main-header {
        background: linear-gradient(135deg, #2d6a4f 0%, #52b788 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-align: center;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    /* Section headers */
    .section-header {
        color: #2d6a4f;
        font-size: 1.5rem;
        font-weight: 600;
        padding: 1rem 0;
        border-bottom: 3px solid #52b788;
        margin-bottom: 1.5rem;
    }
            
    /* Prediction result box */
    .prediction-box {
        background: linear-gradient(135deg, #52b788 0%, #2d6a4f 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin: 2rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        animation: fadeInUp 0.5s ease;
    }
    
    .prediction-box h2 {
        font-size: 2rem;
        margin: 0;
        font-weight: 700;
    }
    
    .prediction-box p {
        font-size: 1.1rem;
        margin-top: 0.5rem;
        opacity: 0.95;
    }
    
    /* Info cards */
    .info-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #52b788;
        margin-bottom: 1rem;
        transition: transform 0.2s;
    }
    
    .info-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
    }
    
    /* Metric styling */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        font-weight: 700;
        color: #2d6a4f;
    }
    
    /* Button styling */
    .stButton>button {
        background: linear-gradient(135deg, #52b788 0%, #2d6a4f 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border: none;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: #f8f9fa;
        border-radius: 10px 10px 0 0;
        padding: 1rem 1.5rem;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #52b788 0%, #2d6a4f 100%);
        color: white;
    }
    
    /* Sidebar styling */           
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d6a4f 0%, #1b4332 100%);
    }
    
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 8px;
        font-weight: 600;
        border-left: 4px solid #52b788;
    }
    
    /* Animation */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Remove extra padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* DataFrame styling */
    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    try:
        rf_model = joblib.load("models/crop_random_model.pkl")
        dt_model = joblib.load("models/crop_tree_model.pkl")
        knn_model = joblib.load("models/crop_knn_model.pkl")
        return {
            "Random Forest": rf_model,
            "Decision Tree": dt_model,
            "KNN": knn_model
        }
    except Exception as e:
        st.error(f"⚠️ Error loading models: {e}")
        return None

st.set_page_config(
    page_title="Sow Smart - Crop Recommendation",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Header
st.markdown('<div class="main-header"><h1>🌾 Sow Smart - Crop Recommendation System</h1></div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("image/crops (1).png",width=150)
    st.title("🌱 Crop Assistant")
    st.markdown("---")
    st.markdown("""
        ### Features
        - 🔮 **Smart Predictions** - AI-powered crop recommendations
        - 📊 **Data Insights** - Visualize your agricultural data
        - 🧠 **Model Selection** - Compare ML algorithms
        - 📚 **Crop Guide** - Comprehensive growing information
    """)
    st.markdown("---")
    st.markdown("### About")
    st.info("This AI system analyzes soil nutrients, weather conditions, and rainfall to recommend optimal crops for maximum yield.")

# Load Models
models = load_models()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict Crop", "📊 Data Insights", "🧠 Model Selection", "📚 Crop Guide"])

# TAB 1: Predict Crop
with tab1:
    st.markdown('<p class="section-header">🌱 Enter Soil & Climate Parameters</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🧪 Soil Nutrients")
        N = st.number_input("Nitrogen (N)", min_value=0, max_value=200, value=90, help="Nitrogen content in soil (kg/ha)")
        P = st.number_input("Phosphorus (P)", min_value=0, max_value=200, value=42, help="Phosphorus content in soil (kg/ha)")
        K = st.number_input("Potassium (K)", min_value=0, max_value=200, value=43, help="Potassium content in soil (kg/ha)")

    with col2:
        st.markdown("#### 🌡️ Climate Factors")
        temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=60.0, value=20.8, step=0.1)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=82.0, step=0.1)
        rainfall = st.number_input("Rainfall (cm)", min_value=0.0, max_value=500.0, value=203.0, step=0.1)

    with col3:
        st.markdown("#### ⚗️ Soil Properties")
        ph = st.number_input("pH Value", min_value=0.0, max_value=14.0, value=6.5, step=0.1, help="Soil acidity/alkalinity level")
        
        st.markdown("#### 🤖 Select Model")
        if models:
            selected_model_name = st.selectbox("Choose Model", options=list(models.keys()), help="Select ML algorithm for prediction")

    st.markdown("<br>", unsafe_allow_html=True)

    # Centered Prediction Button
    col_btn = st.columns([1, 2, 1])
    with col_btn[1]:
        predict_btn = st.button("🚀 PREDICT BEST CROP", use_container_width=True)

    if predict_btn and models:
        data_point = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        selected_model = models[selected_model_name]

        with st.spinner("🔄 Analyzing soil and climate data..."):
            result = selected_model.predict(data_point)[0]
        
        # Display Prediction Result
        st.markdown(f"""
        <div class="prediction-box">
            <h2>🎯 Recommended Crop: {result.upper()}</h2>
            <p>Predicted using {selected_model_name}</p>
        </div>
        """, unsafe_allow_html=True)

        # Input Summary
        st.markdown('<p class="section-header">📋 Input Summary</p>', unsafe_allow_html=True)
        
        col_sum1, col_sum2, col_sum3 = st.columns(3)
        
        with col_sum1:
            st.metric("🧪 N-P-K Ratio", f"{N}-{P}-{K}", help="Nitrogen-Phosphorus-Potassium")
            st.metric("🌡️ Temperature", f"{temperature}°C")
        
        with col_sum2:
            st.metric("💧 Humidity", f"{humidity}%")
            st.metric("⚗️ pH Value", f"{ph}", help="Soil acidity/alkalinity")
        
        with col_sum3:
            st.metric("🌧️ Rainfall", f"{rainfall} cm")
            st.metric("🤖 Model", selected_model_name)

# TAB 2: Data Insights
with tab2:
    st.markdown('<p class="section-header">📊 Dataset Analysis & Visualizations</p>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📁 Upload Your Crop Dataset (CSV)", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        st.markdown("#### 📈 Dataset Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        
        # Histogram
        with col1:
            st.markdown("#### 📉 Feature Distribution")
            feature = st.selectbox("Select Feature", df.columns)

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(df[feature], bins=30, color='#52b788', alpha=0.8, edgecolor='#2d6a4f', linewidth=1.2)
            ax.set_title(f"Distribution of {feature}", fontsize=16, fontweight="bold", color='#2d6a4f')
            ax.set_xlabel(feature, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.grid(alpha=0.3, linestyle='--')
            plt.tight_layout()
            st.pyplot(fig)

        # Box Plot
        with col2:
            st.markdown("#### 📦 Box Plot Analysis")
            if "label" in df.columns:
                feature_box = st.selectbox("Select Feature for Box Plot", [col for col in df.columns if col != "label"])
                
                fig, ax = plt.subplots(figsize=(10, 6))
                df.boxplot(column=feature_box, by="label", ax=ax, patch_artist=True,
                          boxprops=dict(facecolor='#52b788', alpha=0.7),
                          medianprops=dict(color='#2d6a4f', linewidth=2))
                ax.set_title(f"{feature_box} by Crop Type", fontsize=16, fontweight="bold", color='#2d6a4f')
                plt.suptitle('')
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.warning("⚠️ No 'label' column found for crop comparison")

        # Crop Comparison
        if "label" in df.columns:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown('<p class="section-header">🌾 Crop-Specific Comparison</p>', unsafe_allow_html=True)
            
            feature_compare = st.selectbox("Select Feature for Comparison", [col for col in df.columns if col != "label"], key="compare")

            fig, ax = plt.subplots(figsize=(12, 6))
            df.groupby("label")[feature_compare].mean().sort_values().plot(
                kind="barh", ax=ax, color='#52b788', edgecolor='#2d6a4f', linewidth=1.5
            )
            ax.set_title(f"Average {feature_compare} Across Crops", fontsize=16, fontweight="bold", color='#2d6a4f')
            ax.set_xlabel(f"Average {feature_compare}", fontsize=12)
            ax.grid(alpha=0.3, linestyle='--', axis='x')
            plt.tight_layout()
            st.pyplot(fig)
    else:
        st.info("👆 Upload a CSV dataset to explore visualizations and insights")

# TAB 3: Model Selection
with tab3:
    st.markdown('<p class="section-header">⚡ Model Performance Comparison</p>', unsafe_allow_html=True)

    if models:
        accuracy_scores = {
            "Random Forest": 99.3,
            "Decision Tree": 98.4,
            "KNN": 97.0
        }

        # Best Model Highlight
        best_model = max(accuracy_scores, key=accuracy_scores.get)
        best_accuracy = accuracy_scores[best_model]

        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("#### 📊 Accuracy Comparison")
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ['#f4a261' if model == best_model else '#52b788' for model in accuracy_scores.keys()]
            bars = ax.bar(accuracy_scores.keys(), accuracy_scores.values(), color=colors, edgecolor='#2d6a4f', linewidth=2)
            ax.set_title("Model Accuracy Comparison", fontsize=16, fontweight="bold", color='#2d6a4f')
            ax.set_ylabel("Accuracy (%)", fontsize=12)
            ax.set_ylim(95, 100)
            ax.grid(alpha=0.3, linestyle='--', axis='y')
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            st.markdown("#### 🏆 Best Model")
            st.markdown(f"""
            <div class="info-card">
                <h3 style="color: #2d6a4f; margin: 0;">{best_model}</h3>
                <p style="font-size: 2rem; font-weight: bold; color: #f4a261; margin: 0.5rem 0;">{best_accuracy}%</p>
                <p style="margin: 0; opacity: 0.8;">Highest Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 📋 All Models")
            compare_df = pd.DataFrame({
                "Model": list(accuracy_scores.keys()),
                "Accuracy (%)": list(accuracy_scores.values())
            })
            st.dataframe(compare_df, use_container_width=True, hide_index=True)

        # Model Characteristics
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<p class="section-header">📝 Model Characteristics</p>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="info-card">
                <h4 style="color: #2d6a4f;">🌲 Random Forest</h4>
                <ul style="padding-left: 1.5rem;">
                    <li>Best overall accuracy</li>
                    <li>Robust & stable</li>
                    <li>Handles complex patterns</li>
                    <li>Recommended for production</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="info-card">
                <h4 style="color: #2d6a4f;">🌳 Decision Tree</h4>
                <ul style="padding-left: 1.5rem;">
                    <li>Highly interpretable</li>
                    <li>Fast predictions</li>
                    <li>May overfit</li>
                    <li>Good for debugging</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="info-card">
                <h4 style="color: #2d6a4f;">📍 K-Nearest Neighbors</h4>
                <ul style="padding-left: 1.5rem;">
                    <li>Instance-based learning</li>
                    <li>Good for small datasets</li>
                    <li>Slower predictions</li>
                    <li>Sensitive to noise</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

# TAB 4: Crop Guide
with tab4:
    st.markdown('<p class="section-header">📚 Comprehensive Crop Growing Guide</p>', unsafe_allow_html=True)
    
    crop_data = {
        "🌾 Rice": {
            "Climate": "Warm and humid (20-35°C)",
            "Soil": "Clay or loamy soil with good water retention",
            "Rainfall": "150-300 cm annually",
            "pH": "5.5 - 7.0",
            "NPK": "High nitrogen, moderate phosphorus and potassium",
            "Growing Season": "3-6 months"
        },
        "🌽 Maize": {
            "Climate": "Warm weather (18-32°C)",
            "Soil": "Well-drained, fertile loamy soil",
            "Rainfall": "50-100 cm",
            "pH": "5.8 - 7.0",
            "NPK": "High nitrogen, moderate phosphorus and potassium",
            "Growing Season": "3-5 months"
        },
        "🧆 Chickpea": {
            "Climate": "Cool and dry climate (10-30°C)",
            "Soil": "Well-drained loamy to clay soils",
            "Rainfall": "60-90 cm",
            "pH": "6.0 - 9.0",
            "NPK": "Moderate nitrogen, high phosphorus",
            "Growing Season": "3-5 months"
        },
        "🔴 Kidney Beans": {
            "Climate": "Warm climate (15-27°C)",
            "Soil": "Well-drained loamy soil",
            "Rainfall": "40-60 cm",
            "pH": "6.0 - 7.0",
            "NPK": "Moderate nitrogen and phosphorus",
            "Growing Season": "3-4 months"
        },
        "🟡 Pigeon Peas": {
            "Climate": "Tropical climate (20-35°C)",
            "Soil": "Loamy or sandy loam soils",
            "Rainfall": "60-100 cm",
            "pH": "6.0 - 7.5",
            "NPK": "Moderate nitrogen, high phosphorus",
            "Growing Season": "5-6 months"
        },
        "🟤 Moth Beans": {
            "Climate": "Hot and dry (25-35°C)",
            "Soil": "Sandy or loamy soils",
            "Rainfall": "20-40 cm",
            "pH": "7.0 - 8.5",
            "NPK": "Low nitrogen requirement",
            "Growing Season": "3-4 months"
        },
        "🟢 Mung Bean": {
            "Climate": "Warm climate (20-35°C)",
            "Soil": "Loamy soils rich in organic matter",
            "Rainfall": "60-120 cm",
            "pH": "6.2 - 7.2",
            "NPK": "Moderate nitrogen and phosphorus",
            "Growing Season": "2-3 months"
        },
        "⚫ Black Gram": {
            "Climate": "Warm and humid climate (25-35°C)",
            "Soil": "Loamy or clay-loam soils",
            "Rainfall": "60-90 cm",
            "pH": "6.0 - 7.5",
            "NPK": "High phosphorus, moderate nitrogen",
            "Growing Season": "3-4 months"
        },
        "🟠 Lentil": {
            "Climate": "Cool climate (10-25°C)",
            "Soil": "Well-drained loamy soils",
            "Rainfall": "30-45 cm",
            "pH": "6.0 - 8.0",
            "NPK": "Moderate nitrogen, high phosphorus",
            "Growing Season": "3-5 months"
        },
        "🍈 Pomegranate": {
            "Climate": "Hot, dry climate (20-40°C)",
            "Soil": "Well-drained loamy soil",
            "Rainfall": "50-60 cm",
            "pH": "5.5 - 7.0",
            "NPK": "Moderate nitrogen, high potassium",
            "Growing Season": "6-7 months"
        },
        "🍌 Banana": {
            "Climate": "Warm and humid (26-30°C)",
            "Soil": "Loamy soil rich in organic matter",
            "Rainfall": "150-250 cm",
            "pH": "6.5 - 7.5",
            "NPK": "High nitrogen and potassium",
            "Growing Season": "10-12 months"
        },
        "🥭 Mango": {
            "Climate": "Warm tropical climate (24-30°C)",
            "Soil": "Well-drained alluvial soil",
            "Rainfall": "75-250 cm",
            "pH": "5.5 - 7.5",
            "NPK": "Moderate nitrogen, high potassium",
            "Growing Season": "6-11 months"
        },
        "🍇 Grapes": {
            "Climate": "Warm dry climate (15-40°C)",
            "Soil": "Sandy loam or black soil",
            "Rainfall": "50-75 cm",
            "pH": "6.5 - 7.5",
            "NPK": "High nitrogen and potassium",
            "Growing Season": "4-5 months"
        },
        "🍉 Watermelon": {
            "Climate": "Hot climate (25-35°C)",
            "Soil": "Sandy loam soil",
            "Rainfall": "50-75 cm",
            "pH": "6.0 - 6.8",
            "NPK": "Moderate nitrogen, high potassium",
            "Growing Season": "3-4 months"
        },
        "🍈 Muskmelon": {
            "Climate": "Hot and dry (25-35°C)",
            "Soil": "Sandy loam soil",
            "Rainfall": "40-60 cm",
            "pH": "6.0 - 6.7",
            "NPK": "Moderate nitrogen, high potassium",
            "Growing Season": "2.5-3 months"
        },
        "🍎 Apple": {
            "Climate": "Cold climate (0-20°C)",
            "Soil": "Well-drained loamy soil",
            "Rainfall": "100-125 cm",
            "pH": "5.5 - 6.5",
            "NPK": "High potassium",
            "Growing Season": "6-7 months"
        },
        "🍊 Orange": {
            "Climate": "Subtropical climate (15-30°C)",
            "Soil": "Well-drained sandy loam",
            "Rainfall": "75-120 cm",
            "pH": "5.5 - 7.0",
            "NPK": "High nitrogen and potassium",
            "Growing Season": "6-8 months"
        },
        "🍈 Papaya": {
            "Climate": "Warm tropical climate (25-35°C)",
            "Soil": "Sandy loam soil rich in organic matter",
            "Rainfall": "150-200 cm",
            "pH": "6.0 - 6.5",
            "NPK": "High nitrogen",
            "Growing Season": "6-9 months"
        },
        "🥥 Coconut": {
            "Climate": "Humid tropical (20-32°C)",
            "Soil": "Sandy loam soil",
            "Rainfall": "150-250 cm",
            "pH": "5.0 - 8.0",
            "NPK": "High potassium",
            "Growing Season": "12-14 months"
        },
        "☁️ Cotton": {
            "Climate": "Hot climate (21-30°C)",
            "Soil": "Black soil or deep loamy soil",
            "Rainfall": "50-100 cm",
            "pH": "6.0 - 8.0",
            "NPK": "Moderate nitrogen, high phosphorus and potassium",
            "Growing Season": "5-6 months"
        },
        "🧵 Jute": {
            "Climate": "Warm and humid (24-37°C)",
            "Soil": "Alluvial soil",
            "Rainfall": "150-200 cm",
            "pH": "6.4 - 7.2",
            "NPK": "Moderate nitrogen and phosphorus",
            "Growing Season": "4-5 months"
        },
        "☕ Coffee": {
            "Climate": "Cool tropical (15-25°C)",
            "Soil": "Well-drained loamy soil with organic matter",
            "Rainfall": "150-250 cm",
            "pH": "6.0 - 6.5",
            "NPK": "High nitrogen and potassium",
            "Growing Season": "6-11 months"
        }
    }

    # Search functionality
    search_term = st.text_input("🔍 Search for a crop", placeholder="Type crop name...")
    
    filtered_crops = {k: v for k, v in crop_data.items() if search_term.lower() in k.lower()} if search_term else crop_data

    if filtered_crops:
        for crop, info in filtered_crops.items():
            with st.expander(f"{crop}", expanded=False):
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**🌡️ Climate:** {info['Climate']}")
                    st.markdown(f"**🌱 Soil:** {info['Soil']}")
                    st.markdown(f"**🌧️ Rainfall:** {info['Rainfall']}")
                with col2:
                    st.markdown(f"**⚗️ pH Range:** {info['pH']}")
                    st.markdown(f"**🧪 NPK Requirements:** {info['NPK']}")
                    st.markdown(f"**⏱️ Growing Season:** {info['Growing Season']}")
    else:
        st.warning("❌ No crops found matching your search")

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1.5rem;">
    <p style="font-size: 1.2rem; font-weight: 600; color: #2d6a4f; margin-bottom: 0.5rem;">🌾 Sow Smart - Crop Recommendation System</p>
    <p style="margin: 0; opacity: 0.8;">Powered by Machine Learning | Enabling Smarter Farming Decisions</p>
</div>
""", unsafe_allow_html=True)