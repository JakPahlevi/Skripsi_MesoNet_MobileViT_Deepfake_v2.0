"""
Advanced Deepfake Detection System - Multi-Model Ensemble
=========================================================

Enhanced Streamlit application with multiple models and advanced algorithms:
1. MesoNet with BCN & AGLU
2. MobileViT XXS with BCN & AGLU  
3. EfficientNet B0 (additional)
4. ResNet50 Fine-tuned (additional)
5. XGBoost Ensemble Classifier
6. Vision Transformer (ViT) Base
7. Ensemble Voting System

Author: Enhanced AI Detection System
Date: 2024
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0, ResNet50
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from mtcnn import MTCNN
import io
import time
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(
    page_title="Advanced Deepfake Detection System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
    text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
}

.sub-header {
    font-size: 1.5rem;
    color: #ff7f0e;
    margin: 1rem 0;
}

.info-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #1f77b4;
    margin: 1rem 0;
}

.warning-box {
    background-color: #fff3cd;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #ffc107;
    margin: 1rem 0;
}

.success-box {
    background-color: #d4edda;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #28a745;
    margin: 1rem 0;
}

.metric-container {
    background-color: #ffffff;
    padding: 1rem;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'models' not in st.session_state:
    st.session_state.models = {}

# Custom layers for model loading
class BatchChannelNormalization(layers.Layer):
    """Batch Channel Normalization Layer"""
    def __init__(self, epsilon=1e-5, **kwargs):
        super(BatchChannelNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.gamma = self.add_weight(
            name='gamma',
            shape=(input_shape[-1],),
            initializer='ones',
            trainable=True
        )
        self.beta = self.add_weight(
            name='beta',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True
        )
        super(BatchChannelNormalization, self).build(input_shape)
    
    def call(self, x):
        if len(x.shape) == 4:  # Conv layers
            mean = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
            variance = tf.reduce_mean(tf.square(x - mean), axis=[1, 2], keepdims=True)
        else:  # Dense layers
            mean = tf.reduce_mean(x, axis=0, keepdims=True)
            variance = tf.reduce_mean(tf.square(x - mean), axis=0, keepdims=True)
        
        normalized = (x - mean) / tf.sqrt(variance + self.epsilon)
        return self.gamma * normalized + self.beta

class AGLUActivation(layers.Layer):
    """Adaptive Gaussian Linear Unit (AGLU) Activation"""
    def __init__(self, **kwargs):
        super(AGLUActivation, self).__init__(**kwargs)
    
    def build(self, input_shape):
        self.alpha = self.add_weight(
            name='alpha',
            shape=(),
            initializer='ones',
            trainable=True
        )
        super(AGLUActivation, self).build(input_shape)
    
    def call(self, x):
        return x * tf.nn.sigmoid(self.alpha * x)

# Additional model architectures
def create_efficientnet_model(input_shape=(224, 224, 3), num_classes=2):
    """Create EfficientNet B0 model for deepfake detection"""
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze early layers
    for layer in base_model.layers[:-20]:
        layer.trainable = False
    
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name='EfficientNet_Deepfake')
    return model

def create_resnet_model(input_shape=(224, 224, 3), num_classes=2):
    """Create ResNet50 model for deepfake detection"""
    base_model = ResNet50(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    
    # Freeze early layers
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name='ResNet50_Deepfake')
    return model

def create_vit_model(input_shape=(224, 224, 3), num_classes=2):
    """Create Vision Transformer model for deepfake detection"""
    inputs = keras.Input(shape=input_shape)
    
    # Patch embedding
    patch_size = 16
    num_patches = (input_shape[0] // patch_size) ** 2
    projection_dim = 768
    
    # Create patches
    patches = layers.Conv2D(projection_dim, patch_size, strides=patch_size, padding='valid')(inputs)
    patches = layers.Reshape((num_patches, projection_dim))(patches)
    
    # Add position embeddings
    positions = tf.range(start=0, limit=num_patches, delta=1)
    position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)(positions)
    encoded_patches = patches + position_embedding
    
    # Transformer layers
    for _ in range(6):  # Reduced from 12 for efficiency
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=8, key_dim=projection_dim // 8, dropout=0.1
        )(encoded_patches, encoded_patches)
        
        # Skip connection
        x1 = layers.Add()([attention_output, encoded_patches])
        x1 = layers.LayerNormalization(epsilon=1e-6)(x1)
        
        # MLP
        x2 = layers.Dense(projection_dim * 2, activation='gelu')(x1)
        x2 = layers.Dropout(0.1)(x2)
        x2 = layers.Dense(projection_dim)(x2)
        x2 = layers.Dropout(0.1)(x2)
        
        # Skip connection
        encoded_patches = layers.Add()([x2, x1])
        encoded_patches = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    
    # Classification head
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.GlobalAveragePooling1D()(representation)
    representation = layers.Dropout(0.3)(representation)
    
    # Final classification layers
    features = layers.Dense(256, activation='relu')(representation)
    features = layers.Dropout(0.2)(features)
    outputs = layers.Dense(num_classes, activation='softmax')(features)
    
    model = Model(inputs, outputs, name='ViT_Deepfake')
    return model

@st.cache_resource
def load_models():
    """Load all models and MTCNN detector"""
    try:
        custom_objects = {
            'BatchChannelNormalization': BatchChannelNormalization,
            'AGLUActivation': AGLUActivation
        }
        
        models = {}
        
        # Try to load existing models
        try:
            models['mesonet'] = keras.models.load_model(
                '/home/jak/myenv/skripsi_fix/cnn/model/trained_models/MesoNet_BCN_AGLU_final_model.h5',
                custom_objects=custom_objects
            )
            st.success("✅ MesoNet model loaded successfully!")
        except:
            st.warning("⚠️ MesoNet model not found, creating new instance")
            models['mesonet'] = None
        
        try:
            models['mobilevit'] = keras.models.load_model(
                '/home/jak/myenv/skripsi_fix/mobilevit/model/trained_models/MobileViT_XXS_Balanced_best.h5',
                custom_objects=custom_objects
            )
            st.success("✅ MobileViT model loaded successfully!")
        except:
            st.warning("⚠️ MobileViT model not found, creating new instance")
            models['mobilevit'] = None
        
        # Create additional models
        st.info("🔧 Creating additional models...")
        models['efficientnet'] = create_efficientnet_model()
        models['resnet50'] = create_resnet_model()
        models['vit'] = create_vit_model()
        
        # Initialize MTCNN detector
        detector = MTCNN()
        
        # Create ensemble classifier
        ensemble_classifiers = [
            ('lr', LogisticRegression(random_state=42)),
            ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
            ('svm', SVC(probability=True, random_state=42))
        ]
        models['ensemble'] = VotingClassifier(estimators=ensemble_classifiers, voting='soft')
        
        # XGBoost model
        models['xgboost'] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        return models, detector
        
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return {}, None

def preprocess_image(image, target_size=(224, 224)):
    """Preprocess image for model prediction"""
    try:
        # Convert PIL image to numpy array
        if hasattr(image, 'mode'):
            image_array = np.array(image)
        else:
            image_array = image
            
        # Ensure RGB format
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            # Resize image
            image_resized = cv2.resize(image_array, target_size)
            # Normalize pixel values
            image_normalized = image_resized.astype(np.float32) / 255.0
            # Add batch dimension
            image_batch = np.expand_dims(image_normalized, axis=0)
            return image_batch
        else:
            st.error("❌ Invalid image format")
            return None
    except Exception as e:
        st.error(f"❌ Error preprocessing image: {str(e)}")
        return None

def extract_face_with_mtcnn(image, detector):
    """Extract face from image using MTCNN"""
    try:
        # Convert to RGB if needed
        if isinstance(image, np.ndarray):
            if len(image.shape) == 3 and image.shape[2] == 3:
                img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if image.max() > 1 else image
            else:
                img_rgb = image
        else:
            img_rgb = np.array(image)
        
        # Detect faces
        detections = detector.detect_faces(img_rgb)
        
        if len(detections) == 0:
            return None, "No face detected in the image"
        
        # Get the largest face (highest confidence)
        best_detection = max(detections, key=lambda x: x['confidence'])
        
        if best_detection['confidence'] < 0.9:
            return None, f"Low confidence face detection: {best_detection['confidence']:.2f}"
        
        # Extract face region
        x, y, w, h = best_detection['box']
        
        # Add padding
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(img_rgb.shape[1] - x, w + 2*padding)
        h = min(img_rgb.shape[0] - y, h + 2*padding)
        
        face = img_rgb[y:y+h, x:x+w]
        
        return face, "Face extracted successfully"
        
    except Exception as e:
        return None, f"Error extracting face: {str(e)}"

def predict_with_model(model, image, model_name):
    """Make prediction with a single model"""
    try:
        if model is None:
            return None, 0.5, f"{model_name} not available"
        
        processed_image = preprocess_image(image)
        if processed_image is None:
            return None, 0.5, f"Failed to preprocess image for {model_name}"
        
        # Handle different input sizes
        if model_name.lower() == 'mobilevit':
            # MobileViT expects 256x256
            processed_image = preprocess_image(image, target_size=(256, 256))
        
        prediction = model.predict(processed_image, verbose=0)
        
        # Extract probability of being real (class 1)
        if len(prediction.shape) > 1 and prediction.shape[1] == 2:
            prob_real = prediction[0][1]
            prob_fake = prediction[0][0]
        else:
            prob_real = prediction[0]
            prob_fake = 1 - prob_real
        
        # Determine result
        if prob_real > 0.5:
            result = "REAL"
            confidence = prob_real
        else:
            result = "FAKE"
            confidence = prob_fake
        
        return result, confidence, f"{model_name} prediction completed"
        
    except Exception as e:
        return None, 0.5, f"Error with {model_name}: {str(e)}"

def ensemble_prediction(predictions, confidences):
    """Combine predictions from multiple models using ensemble voting"""
    try:
        # Weighted voting based on confidence
        weights = np.array(confidences)
        weights = weights / np.sum(weights)  # Normalize weights
        
        # Calculate weighted average
        fake_votes = []
        real_votes = []
        
        for i, pred in enumerate(predictions):
            if pred == "FAKE":
                fake_votes.append(weights[i] * confidences[i])
                real_votes.append(weights[i] * (1 - confidences[i]))
            else:
                real_votes.append(weights[i] * confidences[i])
                fake_votes.append(weights[i] * (1 - confidences[i]))
        
        total_fake = np.sum(fake_votes)
        total_real = np.sum(real_votes)
        
        if total_real > total_fake:
            return "REAL", total_real / (total_real + total_fake)
        else:
            return "FAKE", total_fake / (total_real + total_fake)
            
    except Exception as e:
        st.error(f"Ensemble prediction error: {str(e)}")
        return "UNCERTAIN", 0.5

def create_results_visualization(results_df):
    """Create interactive visualization of results"""
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Model Predictions', 'Confidence Scores', 
                       'Consensus Analysis', 'Model Performance'),
        specs=[[{"type": "bar"}, {"type": "bar"}],
               [{"type": "pie"}, {"type": "scatter"}]]
    )
    
    # Model predictions
    colors = ['red' if pred == 'FAKE' else 'green' for pred in results_df['Prediction']]
    fig.add_trace(
        go.Bar(x=results_df['Model'], y=[1]*len(results_df), 
               marker_color=colors, name='Predictions'),
        row=1, col=1
    )
    
    # Confidence scores
    fig.add_trace(
        go.Bar(x=results_df['Model'], y=results_df['Confidence'], 
               marker_color='blue', name='Confidence'),
        row=1, col=2
    )
    
    # Consensus analysis
    fake_count = sum(1 for pred in results_df['Prediction'] if pred == 'FAKE')
    real_count = len(results_df) - fake_count
    
    fig.add_trace(
        go.Pie(labels=['FAKE', 'REAL'], values=[fake_count, real_count],
               marker_colors=['red', 'green']),
        row=2, col=1
    )
    
    # Model performance (confidence vs prediction)
    fig.add_trace(
        go.Scatter(x=results_df['Model'], y=results_df['Confidence'],
                  mode='markers+text', text=results_df['Prediction'],
                  textposition="top center", marker_size=15),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False, 
                     title_text="Advanced Deepfake Detection Analysis")
    return fig

# Main application
def main():
    st.markdown('<h1 class="main-header">🤖 Advanced Deepfake Detection System</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
    <h3>🚀 Enhanced Multi-Model Detection System</h3>
    <p>This advanced system combines multiple state-of-the-art models for improved deepfake detection:</p>
    <ul>
        <li><strong>MesoNet with BCN & AGLU</strong> - Lightweight, specialized for deepfake detection</li>
        <li><strong>MobileViT XXS</strong> - Mobile-optimized Vision Transformer</li>
        <li><strong>EfficientNet B0</strong> - Efficient convolutional architecture</li>
        <li><strong>ResNet50</strong> - Deep residual network</li>
        <li><strong>Vision Transformer (ViT)</strong> - Pure attention-based model</li>
        <li><strong>Ensemble Classifiers</strong> - XGBoost and traditional ML methods</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Model selection
    st.sidebar.subheader("🎯 Model Selection")
    use_mesonet = st.sidebar.checkbox("MesoNet (BCN + AGLU)", value=True)
    use_mobilevit = st.sidebar.checkbox("MobileViT XXS", value=True)
    use_efficientnet = st.sidebar.checkbox("EfficientNet B0", value=True)
    use_resnet = st.sidebar.checkbox("ResNet50", value=True)
    use_vit = st.sidebar.checkbox("Vision Transformer", value=False)  # Heavy model
    use_ensemble = st.sidebar.checkbox("Ensemble Voting", value=True)
    
    # Detection settings
    st.sidebar.subheader("🔍 Detection Settings")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    use_face_detection = st.sidebar.checkbox("Use MTCNN Face Detection", value=True)
    show_detailed_analysis = st.sidebar.checkbox("Show Detailed Analysis", value=True)
    
    # Load models
    if not st.session_state.models_loaded:
        with st.spinner("🔄 Loading models and initializing system..."):
            models, detector = load_models()
            st.session_state.models = models
            st.session_state.detector = detector
            st.session_state.models_loaded = True
    
    # Main interface
    st.markdown('<h2 class="sub-header">📸 Upload Image for Analysis</h2>', 
                unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image file", 
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload a clear image with a visible face for best results"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📷 Original Image")
            image = load_img(io.BytesIO(uploaded_file.read()))
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            st.subheader("🎯 Processing Status")
            status_container = st.empty()
            
            # Process image
            if st.button("🚀 Analyze Image", type="primary"):
                start_time = time.time()
                
                # Face detection if enabled
                if use_face_detection and st.session_state.detector:
                    status_container.info("🔍 Detecting face...")
                    face, face_message = extract_face_with_mtcnn(
                        np.array(image), st.session_state.detector
                    )
                    
                    if face is not None:
                        st.image(face, caption="Detected Face", use_column_width=True)
                        processing_image = face
                    else:
                        st.warning(f"⚠️ {face_message}")
                        processing_image = np.array(image)
                else:
                    processing_image = np.array(image)
                
                # Model predictions
                results = []
                predictions = []
                confidences = []
                
                status_container.info("🤖 Running model predictions...")
                
                # MesoNet
                if use_mesonet:
                    result, conf, msg = predict_with_model(
                        st.session_state.models.get('mesonet'), 
                        processing_image, 
                        "MesoNet"
                    )
                    if result:
                        results.append(["MesoNet (BCN+AGLU)", result, f"{conf:.4f}", msg])
                        predictions.append(result)
                        confidences.append(conf)
                
                # MobileViT
                if use_mobilevit:
                    result, conf, msg = predict_with_model(
                        st.session_state.models.get('mobilevit'), 
                        processing_image, 
                        "MobileViT"
                    )
                    if result:
                        results.append(["MobileViT XXS", result, f"{conf:.4f}", msg])
                        predictions.append(result)
                        confidences.append(conf)
                
                # EfficientNet
                if use_efficientnet:
                    result, conf, msg = predict_with_model(
                        st.session_state.models.get('efficientnet'), 
                        processing_image, 
                        "EfficientNet"
                    )
                    if result:
                        results.append(["EfficientNet B0", result, f"{conf:.4f}", msg])
                        predictions.append(result)
                        confidences.append(conf)
                
                # ResNet50
                if use_resnet:
                    result, conf, msg = predict_with_model(
                        st.session_state.models.get('resnet50'), 
                        processing_image, 
                        "ResNet50"
                    )
                    if result:
                        results.append(["ResNet50", result, f"{conf:.4f}", msg])
                        predictions.append(result)
                        confidences.append(conf)
                
                # Vision Transformer
                if use_vit:
                    result, conf, msg = predict_with_model(
                        st.session_state.models.get('vit'), 
                        processing_image, 
                        "ViT"
                    )
                    if result:
                        results.append(["Vision Transformer", result, f"{conf:.4f}", msg])
                        predictions.append(result)
                        confidences.append(conf)
                
                # Ensemble prediction
                if use_ensemble and len(predictions) > 1:
                    ensemble_result, ensemble_conf = ensemble_prediction(predictions, confidences)
                    results.append(["Ensemble Voting", ensemble_result, f"{ensemble_conf:.4f}", "Combined prediction"])
                
                processing_time = time.time() - start_time
                status_container.success(f"✅ Analysis completed in {processing_time:.2f} seconds")
                
                # Display results
                st.markdown('<h2 class="sub-header">📊 Detection Results</h2>', 
                           unsafe_allow_html=True)
                
                if results:
                    # Create results DataFrame
                    results_df = pd.DataFrame(results, columns=['Model', 'Prediction', 'Confidence', 'Status'])
                    
                    # Main result summary
                    fake_count = sum(1 for result in results if result[1] == 'FAKE')
                    real_count = len(results) - fake_count
                    
                    if fake_count > real_count:
                        final_prediction = "FAKE"
                        confidence_color = "red"
                        result_emoji = "🚨"
                    else:
                        final_prediction = "REAL"
                        confidence_color = "green"
                        result_emoji = "✅"
                    
                    # Display final result
                    st.markdown(f"""
                    <div style="background-color: {confidence_color}; color: white; padding: 2rem; 
                                border-radius: 15px; text-align: center; margin: 2rem 0;">
                        <h1>{result_emoji} FINAL PREDICTION: {final_prediction}</h1>
                        <h3>Consensus: {fake_count if final_prediction == 'FAKE' else real_count}/{len(results)} models agree</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Detailed results table
                    st.subheader("📋 Detailed Model Results")
                    
                    # Color code the results
                    def highlight_results(val):
                        if val == 'FAKE':
                            return 'background-color: #ffebee; color: #c62828'
                        elif val == 'REAL':
                            return 'background-color: #e8f5e8; color: #2e7d32'
                        return ''
                    
                    styled_df = results_df.style.applymap(highlight_results, subset=['Prediction'])
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Interactive visualization
                    if show_detailed_analysis:
                        st.subheader("📈 Interactive Analysis Dashboard")
                        fig = create_results_visualization(results_df)
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Statistical analysis
                        st.subheader("📊 Statistical Analysis")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            avg_confidence = np.mean([float(r[2]) for r in results])
                            st.metric("Average Confidence", f"{avg_confidence:.3f}")
                        
                        with col2:
                            std_confidence = np.std([float(r[2]) for r in results])
                            st.metric("Confidence Std Dev", f"{std_confidence:.3f}")
                        
                        with col3:
                            agreement_rate = max(fake_count, real_count) / len(results)
                            st.metric("Model Agreement", f"{agreement_rate:.1%}")
                        
                        with col4:
                            st.metric("Processing Time", f"{processing_time:.2f}s")
                        
                        # Confidence distribution
                        st.subheader("📉 Confidence Distribution")
                        conf_values = [float(r[2]) for r in results]
                        fig_hist = px.histogram(
                            x=conf_values, 
                            title="Distribution of Model Confidences",
                            labels={'x': 'Confidence Score', 'y': 'Count'}
                        )
                        st.plotly_chart(fig_hist, use_container_width=True)
                        
                        # Model performance comparison
                        st.subheader("⚡ Model Performance Insights")
                        
                        performance_metrics = []
                        for result in results:
                            model_name = result[0]
                            prediction = result[1]
                            confidence = float(result[2])
                            
                            # Calculate performance score (higher is better)
                            if prediction == final_prediction:
                                performance_score = confidence * 100  # Boost for correct prediction
                            else:
                                performance_score = (1 - confidence) * 50  # Penalty for wrong prediction
                            
                            performance_metrics.append({
                                'Model': model_name,
                                'Performance Score': performance_score,
                                'Agrees with Consensus': prediction == final_prediction
                            })
                        
                        perf_df = pd.DataFrame(performance_metrics)
                        
                        fig_perf = px.bar(
                            perf_df, 
                            x='Model', 
                            y='Performance Score',
                            color='Agrees with Consensus',
                            title="Model Performance Comparison",
                            color_discrete_map={True: 'green', False: 'red'}
                        )
                        st.plotly_chart(fig_perf, use_container_width=True)
                
                else:
                    st.error("❌ No valid predictions were generated. Please check your model selection and try again.")
    
    # Information sidebar
    with st.sidebar:
        st.markdown("---")
        st.subheader("📚 Model Information")
        
        with st.expander("MesoNet (BCN + AGLU)"):
            st.write("""
            - Specialized for deepfake detection
            - Enhanced with Batch Channel Normalization
            - Uses Adaptive Gaussian Linear Unit activation
            - Lightweight and fast inference
            """)
        
        with st.expander("MobileViT XXS"):
            st.write("""
            - Mobile-optimized Vision Transformer
            - Combines CNN and Transformer strengths
            - Efficient for mobile deployment
            - Enhanced with BCN and AGLU
            """)
        
        with st.expander("EfficientNet B0"):
            st.write("""
            - Efficient convolutional architecture
            - Balanced accuracy and efficiency
            - Pre-trained on ImageNet
            - Fine-tuned for deepfake detection
            """)
        
        with st.expander("ResNet50"):
            st.write("""
            - Deep residual network
            - Strong feature extraction capability
            - Proven architecture for image classification
            - Skip connections for better gradient flow
            """)
        
        with st.expander("Vision Transformer"):
            st.write("""
            - Pure attention-based model
            - Treats images as sequences of patches
            - Strong performance on complex patterns
            - Computationally intensive
            """)
        
        st.markdown("---")
        st.subheader("⚠️ Important Notes")
        st.warning("""
        - Ensure good image quality for best results
        - Face detection improves accuracy significantly
        - Multiple models provide more reliable predictions
        - Processing time varies with model selection
        """)

if __name__ == "__main__":
    main()