"""
Enhanced Video Deepfake Detection System
========================================

Aplikasi khusus untuk deteksi deepfake pada video dengan optimasi MesoNet dan MobileViT
menggunakan algoritma enhancement dan teknik stabilisasi hasil.
"""

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import tempfile
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Try to import required libraries
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    st.warning("⚠️ TensorFlow not available. Running in demo mode.")

try:
    from mtcnn import MTCNN
    MTCNN_AVAILABLE = True
except ImportError:
    MTCNN_AVAILABLE = False

try:
    from scipy import ndimage
    from scipy.signal import savgol_filter
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# Page config
st.set_page_config(
    page_title="Enhanced Video Deepfake Detection",
    page_icon="🎬",
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

.algorithm-box {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

.enhancement-box {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

.stability-box {
    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 15px;
    margin: 1rem 0;
    box-shadow: 0 4px 8px rgba(0,0,0,0.2);
}

.video-stats {
    background-color: #f8f9fa;
    padding: 1rem;
    border-radius: 10px;
    border-left: 5px solid #28a745;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'detector' not in st.session_state:
    st.session_state.detector = None
if 'mesonet_model' not in st.session_state:
    st.session_state.mesonet_model = None
if 'mobilevit_model' not in st.session_state:
    st.session_state.mobilevit_model = None

# Enhanced Model Classes
if TF_AVAILABLE:
    class EnhancedBatchChannelNormalization(layers.Layer):
        """Enhanced Batch Channel Normalization with adaptive scaling"""
        def __init__(self, epsilon=1e-5, momentum=0.99, **kwargs):
            super(EnhancedBatchChannelNormalization, self).__init__(**kwargs)
            self.epsilon = epsilon
            self.momentum = momentum
        
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
            self.moving_mean = self.add_weight(
                name='moving_mean',
                shape=(input_shape[-1],),
                initializer='zeros',
                trainable=False
            )
            self.moving_variance = self.add_weight(
                name='moving_variance',
                shape=(input_shape[-1],),
                initializer='ones',
                trainable=False
            )
            super(EnhancedBatchChannelNormalization, self).build(input_shape)
        
        def call(self, x, training=None):
            if training:
                if len(x.shape) == 4:
                    mean = tf.reduce_mean(x, axis=[0, 1, 2])
                    variance = tf.reduce_mean(tf.square(x - mean), axis=[0, 1, 2])
                else:
                    mean = tf.reduce_mean(x, axis=0)
                    variance = tf.reduce_mean(tf.square(x - mean), axis=0)
                
                # Update moving averages
                self.moving_mean.assign(self.momentum * self.moving_mean + (1 - self.momentum) * mean)
                self.moving_variance.assign(self.momentum * self.moving_variance + (1 - self.momentum) * variance)
                
                normalized = (x - mean) / tf.sqrt(variance + self.epsilon)
            else:
                normalized = (x - self.moving_mean) / tf.sqrt(self.moving_variance + self.epsilon)
            
            return self.gamma * normalized + self.beta

    class AdaptiveAGLUActivation(layers.Layer):
        """Adaptive AGLU with learnable parameters per channel"""
        def __init__(self, **kwargs):
            super(AdaptiveAGLUActivation, self).__init__(**kwargs)
        
        def build(self, input_shape):
            self.alpha = self.add_weight(
                name='alpha',
                shape=(input_shape[-1],),
                initializer=tf.constant_initializer(1.0),
                trainable=True
            )
            self.beta = self.add_weight(
                name='beta',
                shape=(input_shape[-1],),
                initializer=tf.constant_initializer(0.5),
                trainable=True
            )
            super(AdaptiveAGLUActivation, self).build(input_shape)
        
        def call(self, x):
            return x * tf.nn.sigmoid(self.alpha * x + self.beta)

# Enhanced algorithms for stability and consistency
class TemporalStabilizer:
    """Algoritma untuk stabilisasi temporal pada video"""
    
    def __init__(self, window_size=5, threshold=0.3):
        self.window_size = window_size
        self.threshold = threshold
        self.prediction_history = []
        self.confidence_history = []
    
    def add_prediction(self, prediction, confidence):
        """Tambahkan prediksi baru ke history"""
        self.prediction_history.append(prediction)
        self.confidence_history.append(confidence)
        
        # Keep only recent predictions
        if len(self.prediction_history) > self.window_size:
            self.prediction_history.pop(0)
            self.confidence_history.pop(0)
    
    def get_stabilized_prediction(self):
        """Dapatkan prediksi yang distabilkan"""
        if len(self.prediction_history) < 3:
            return self.prediction_history[-1] if self.prediction_history else None
        
        # Temporal smoothing menggunakan weighted average
        weights = np.exp(np.linspace(-1, 0, len(self.prediction_history)))
        weights = weights / np.sum(weights)
        
        # Weighted prediction
        weighted_pred = np.average(self.prediction_history, weights=weights)
        weighted_conf = np.average(self.confidence_history, weights=weights)
        
        # Consistency check
        recent_variance = np.var(self.prediction_history[-3:])
        if recent_variance < self.threshold:
            return weighted_pred, weighted_conf, "STABLE"
        else:
            return weighted_pred, weighted_conf, "UNSTABLE"

class AdaptiveThresholding:
    """Algoritma adaptive thresholding berdasarkan karakteristik video"""
    
    def __init__(self):
        self.frame_qualities = []
        self.lighting_conditions = []
        self.base_threshold = 0.5
    
    def analyze_frame_quality(self, frame):
        """Analisis kualitas frame"""
        # Hitung sharpness (Laplacian variance)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Hitung lighting condition
        brightness = np.mean(gray)
        
        self.frame_qualities.append(sharpness)
        self.lighting_conditions.append(brightness)
        
        return sharpness, brightness
    
    def get_adaptive_threshold(self, confidence, frame_quality, lighting):
        """Dapatkan threshold yang adaptif"""
        # Base threshold adjustment berdasarkan kualitas frame
        quality_factor = min(frame_quality / 100.0, 1.0)  # Normalize
        lighting_factor = 1.0 - abs(lighting - 128) / 128.0  # Optimal around 128
        
        # Adaptive threshold
        adapted_threshold = self.base_threshold * (0.8 + 0.2 * quality_factor * lighting_factor)
        
        return adapted_threshold

class EnsembleOptimizer:
    """Algoritma optimasi ensemble untuk MesoNet dan MobileViT"""
    
    def __init__(self):
        self.mesonet_history = []
        self.mobilevit_history = []
        self.ensemble_history = []
        self.performance_tracker = {
            'mesonet_weight': 0.6,
            'mobilevit_weight': 0.4,
            'confidence_threshold': 0.5
        }
    
    def update_weights(self, mesonet_pred, mobilevit_pred, mesonet_conf, mobilevit_conf):
        """Update bobot berdasarkan performa historis"""
        self.mesonet_history.append((mesonet_pred, mesonet_conf))
        self.mobilevit_history.append((mobilevit_pred, mobilevit_conf))
        
        # Keep recent history only
        if len(self.mesonet_history) > 20:
            self.mesonet_history.pop(0)
            self.mobilevit_history.pop(0)
        
        # Calculate average confidence
        if len(self.mesonet_history) >= 5:
            avg_mesonet_conf = np.mean([conf for _, conf in self.mesonet_history[-10:]])
            avg_mobilevit_conf = np.mean([conf for _, conf in self.mobilevit_history[-10:]])
            
            # Adaptive weight calculation
            total_conf = avg_mesonet_conf + avg_mobilevit_conf
            if total_conf > 0:
                self.performance_tracker['mesonet_weight'] = avg_mesonet_conf / total_conf
                self.performance_tracker['mobilevit_weight'] = avg_mobilevit_conf / total_conf
    
    def get_ensemble_prediction(self, mesonet_pred, mobilevit_pred, mesonet_conf, mobilevit_conf):
        """Dapatkan prediksi ensemble yang dioptimasi"""
        # Update weights
        self.update_weights(mesonet_pred, mobilevit_pred, mesonet_conf, mobilevit_conf)
        
        # Weighted ensemble
        w1 = self.performance_tracker['mesonet_weight']
        w2 = self.performance_tracker['mobilevit_weight']
        
        ensemble_pred = w1 * mesonet_pred + w2 * mobilevit_pred
        ensemble_conf = w1 * mesonet_conf + w2 * mobilevit_conf
        
        # Consistency boost
        if abs(mesonet_pred - mobilevit_pred) < 0.2:  # Models agree
            ensemble_conf *= 1.1  # Boost confidence
        else:  # Models disagree
            ensemble_conf *= 0.9  # Reduce confidence
        
        ensemble_conf = min(ensemble_conf, 1.0)  # Cap at 1.0
        
        return ensemble_pred, ensemble_conf, w1, w2

# Dummy models for demo
class DummyEnhancedModel:
    def __init__(self, name, base_accuracy=0.75):
        self.name = name
        self.base_accuracy = base_accuracy
        self.frame_count = 0
    
    def predict(self, image, verbose=0):
        self.frame_count += 1
        
        # Simulate temporal consistency
        seed = hash(str(image.tobytes())) % 2**32
        np.random.seed(seed + self.frame_count)
        
        if self.name == "MesoNet":
            # MesoNet simulation with slight bias towards fake detection
            fake_prob = np.random.beta(2, 1.8) * 0.85 + 0.1
        else:  # MobileViT
            # MobileViT simulation with balanced detection
            fake_prob = np.random.beta(1.5, 1.5) * 0.8 + 0.15
        
        # Add some temporal stability
        if self.frame_count > 1:
            fake_prob = fake_prob * 0.7 + getattr(self, 'last_pred', 0.5) * 0.3
        
        self.last_pred = fake_prob
        real_prob = 1 - fake_prob
        
        return np.array([[fake_prob, real_prob]])

@st.cache_resource
def load_enhanced_models():
    """Load enhanced MesoNet and MobileViT models"""
    models = {}
    detector = None
    
    if TF_AVAILABLE:
        # Custom objects for model loading
        custom_objects = {
            'EnhancedBatchChannelNormalization': EnhancedBatchChannelNormalization,
            'AdaptiveAGLUActivation': AdaptiveAGLUActivation,
            'BatchChannelNormalization': EnhancedBatchChannelNormalization,  # Fallback
            'AGLUActivation': AdaptiveAGLUActivation,  # Fallback
            # Add any other custom objects that might be in your models
            'batch_channel_normalization_5': EnhancedBatchChannelNormalization,
        }
        
        # Try to load existing models with error handling
        mesonet_path = '/home/jak/myenv/skripsi_fix/cnn/model/trained_models/MesoNet_BCN_AGLU_final_model.h5'
        mobilevit_path = '/home/jak/myenv/skripsi_fix/mobilevit/model/trained_models/MobileViT_XXS_Balanced_best.h5'
        
        # Try MesoNet
        try:
            if os.path.exists(mesonet_path):
                models['mesonet'] = keras.models.load_model(mesonet_path, 
                                                          custom_objects=custom_objects,
                                                          compile=False)
                st.success("✅ Enhanced MesoNet loaded successfully!")
            else:
                models['mesonet'] = DummyEnhancedModel("MesoNet", 0.78)
                st.info("🔧 Using Enhanced MesoNet demo model (file not found)")
        except Exception as e:
            models['mesonet'] = DummyEnhancedModel("MesoNet", 0.78)
            st.warning(f"⚠️ Using Enhanced MesoNet demo: Model loading failed - {str(e)[:100]}...")
        
        # Try MobileViT
        try:
            if os.path.exists(mobilevit_path):
                models['mobilevit'] = keras.models.load_model(mobilevit_path, 
                                                            custom_objects=custom_objects,
                                                            compile=False)
                st.success("✅ Enhanced MobileViT loaded successfully!")
            else:
                models['mobilevit'] = DummyEnhancedModel("MobileViT", 0.82)
                st.info("🔧 Using Enhanced MobileViT demo model (file not found)")
        except Exception as e:
            models['mobilevit'] = DummyEnhancedModel("MobileViT", 0.82)
            st.warning(f"⚠️ Using Enhanced MobileViT demo: Model loading failed - {str(e)[:100]}...")
    else:
        models['mesonet'] = DummyEnhancedModel("MesoNet", 0.78)
        models['mobilevit'] = DummyEnhancedModel("MobileViT", 0.82)
        st.info("🔧 Running in demo mode with simulated enhanced models")
    
    # Initialize MTCNN detector
    if MTCNN_AVAILABLE:
        try:
            detector = MTCNN()
            st.success("✅ Enhanced MTCNN face detector initialized!")
        except Exception as e:
            st.warning(f"⚠️ MTCNN initialization failed: {str(e)}")
            detector = None
    else:
        st.info("🔧 MTCNN not available - using basic face detection")
        detector = None
    
    return models, detector

def extract_frames_from_video(video_file, max_frames=30, skip_frames=5):
    """Extract frames dari video untuk analisis"""
    frames = []
    frame_indices = []
    
    # Save uploaded video to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(video_file.read())
        tmp_path = tmp_file.name
    
    try:
        cap = cv2.VideoCapture(tmp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        st.info(f"📹 Video info: {total_frames} frames, {fps:.1f} FPS")
        
        frame_count = 0
        extracted_count = 0
        
        while cap.isOpened() and extracted_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames untuk sampling
            if frame_count % skip_frames == 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                frame_indices.append(frame_count)
                extracted_count += 1
            
            frame_count += 1
        
        cap.release()
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return frames, frame_indices, total_frames, fps
    
    except Exception as e:
        st.error(f"❌ Error processing video: {str(e)}")
        return [], [], 0, 0

def preprocess_frame(frame, target_size=(224, 224)):
    """Preprocess frame untuk model prediction"""
    try:
        # Resize dan normalize
        frame_resized = cv2.resize(frame, target_size)
        frame_normalized = frame_resized.astype(np.float32) / 255.0
        frame_batch = np.expand_dims(frame_normalized, axis=0)
        return frame_batch
    except Exception as e:
        st.error(f"❌ Error preprocessing frame: {str(e)}")
        return None

def extract_face_from_frame(frame, detector=None):
    """Extract face from frame"""
    try:
        if detector and MTCNN_AVAILABLE:
            detections = detector.detect_faces(frame)
            
            if len(detections) > 0:
                # Get best detection
                best_detection = max(detections, key=lambda x: x['confidence'])
                
                if best_detection['confidence'] > 0.9:
                    x, y, w, h = best_detection['box']
                    
                    # Add padding
                    padding = 20
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(frame.shape[1] - x, w + 2*padding)
                    h = min(frame.shape[0] - y, h + 2*padding)
                    
                    face = frame[y:y+h, x:x+w]
                    return face, f"Face detected (conf: {best_detection['confidence']:.2f})"
        
        # Fallback: center crop
        h, w = frame.shape[:2]
        size = min(h, w)
        start_h = (h - size) // 2
        start_w = (w - size) // 2
        
        face = frame[start_h:start_h+size, start_w:start_w+size]
        return face, "Center crop used"
        
    except Exception as e:
        return frame, f"Error in face extraction: {str(e)}"

def predict_frame_enhanced(models, frame, stabilizer, adaptive_thresh, ensemble_opt):
    """Enhanced prediction untuk single frame"""
    try:
        # Preprocess frame
        mesonet_input = preprocess_frame(frame, (224, 224))
        mobilevit_input = preprocess_frame(frame, (256, 256))
        
        if mesonet_input is None or mobilevit_input is None:
            return None
        
        # Model predictions
        mesonet_pred = models['mesonet'].predict(mesonet_input, verbose=0)
        mobilevit_pred = models['mobilevit'].predict(mobilevit_input, verbose=0)
        
        # Extract probabilities - handle different output formats
        if len(mesonet_pred.shape) > 1 and mesonet_pred.shape[1] >= 2:
            mesonet_prob = float(mesonet_pred[0][1])  # Real probability
            mesonet_conf = float(max(mesonet_pred[0]))
        else:
            mesonet_prob = float(mesonet_pred[0]) if len(mesonet_pred.shape) == 1 else float(mesonet_pred[0][0])
            mesonet_conf = abs(mesonet_prob - 0.5) + 0.5  # Convert to confidence
            if mesonet_prob < 0.5:
                mesonet_prob = 1 - mesonet_prob  # Convert to real probability
        
        if len(mobilevit_pred.shape) > 1 and mobilevit_pred.shape[1] >= 2:
            mobilevit_prob = float(mobilevit_pred[0][1])  # Real probability  
            mobilevit_conf = float(max(mobilevit_pred[0]))
        else:
            mobilevit_prob = float(mobilevit_pred[0]) if len(mobilevit_pred.shape) == 1 else float(mobilevit_pred[0][0])
            mobilevit_conf = abs(mobilevit_prob - 0.5) + 0.5  # Convert to confidence
            if mobilevit_prob < 0.5:
                mobilevit_prob = 1 - mobilevit_prob  # Convert to real probability
        
        # Ensemble optimization
        ensemble_prob, ensemble_conf, w1, w2 = ensemble_opt.get_ensemble_prediction(
            mesonet_prob, mobilevit_prob, mesonet_conf, mobilevit_conf
        )
        
        # Temporal stabilization
        stabilizer.add_prediction(ensemble_prob, ensemble_conf)
        stabilized_result = stabilizer.get_stabilized_prediction()
        
        # Handle stabilizer return value
        if isinstance(stabilized_result, tuple) and len(stabilized_result) >= 3:
            stabilized_prob, stabilized_conf, stability = stabilized_result
        else:
            stabilized_prob = stabilized_result if stabilized_result is not None else ensemble_prob
            stabilized_conf = ensemble_conf
            stability = "STABLE"
        
        # Adaptive thresholding
        frame_quality, lighting = adaptive_thresh.analyze_frame_quality(frame)
        adapted_threshold = adaptive_thresh.get_adaptive_threshold(
            stabilized_conf, frame_quality, lighting
        )
        
        # Final prediction
        final_pred = "REAL" if stabilized_prob > adapted_threshold else "FAKE"
        
        return {
            'mesonet': {'prob': mesonet_prob, 'conf': mesonet_conf},
            'mobilevit': {'prob': mobilevit_prob, 'conf': mobilevit_conf},
            'ensemble': {'prob': ensemble_prob, 'conf': ensemble_conf, 'weights': (w1, w2)},
            'stabilized': {'prob': stabilized_prob, 'conf': stabilized_conf, 'stability': stability},
            'adaptive': {'threshold': adapted_threshold, 'quality': frame_quality, 'lighting': lighting},
            'final': {'prediction': final_pred, 'confidence': stabilized_conf}
        }
        
    except Exception as e:
        print(f"Error in frame prediction: {str(e)}")  # Use print instead of st.error
        return None

def analyze_video_enhanced(models, frames, detector=None):
    """Enhanced video analysis dengan algoritma stabilisasi"""
    
    # Initialize enhancement algorithms
    stabilizer = TemporalStabilizer(window_size=7, threshold=0.25)
    adaptive_thresh = AdaptiveThresholding()
    ensemble_opt = EnsembleOptimizer()
    
    results = []
    face_frames = []
    
    # Simple progress tracking without streamlit dependency
    total_frames = len(frames)
    
    for i, frame in enumerate(frames):
        print(f"Processing frame {i+1}/{total_frames}...")
        
        # Extract face
        face, face_msg = extract_face_from_frame(frame, detector)
        face_frames.append(face)
        
        # Enhanced prediction
        result = predict_frame_enhanced(models, face, stabilizer, adaptive_thresh, ensemble_opt)
        
        if result:
            result_entry = {
                'frame_idx': i,
                'face_extraction': face_msg
            }
            result_entry.update(result)  # Safely merge the result dictionary
            results.append(result_entry)
    
    print("Video analysis completed!")
    
    return results, face_frames

def create_enhanced_visualizations(results):
    """Create enhanced visualizations untuk video analysis"""
    
    if not results:
        st.error("❌ No results to visualize")
        return
    
    # Extract data for plotting
    frames = [r['frame_idx'] for r in results]
    mesonet_probs = [r['mesonet']['prob'] for r in results]
    mobilevit_probs = [r['mobilevit']['prob'] for r in results]
    ensemble_probs = [r['ensemble']['prob'] for r in results]
    stabilized_probs = [r['stabilized']['prob'] for r in results]
    confidences = [r['final']['confidence'] for r in results]
    thresholds = [r['adaptive']['threshold'] for r in results]
    
    # Create comprehensive visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Model Predictions Comparison
    ax1.plot(frames, mesonet_probs, 'b-', label='MesoNet', linewidth=2, alpha=0.7)
    ax1.plot(frames, mobilevit_probs, 'r-', label='MobileViT', linewidth=2, alpha=0.7)
    ax1.plot(frames, ensemble_probs, 'g-', label='Ensemble', linewidth=2)
    ax1.plot(frames, stabilized_probs, 'm-', label='Stabilized', linewidth=3)
    ax1.axhline(y=0.5, color='k', linestyle='--', alpha=0.5, label='Base Threshold')
    ax1.set_title('Model Predictions Over Time', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Frame Index')
    ax1.set_ylabel('Real Probability')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Confidence and Adaptive Threshold
    ax2.plot(frames, confidences, 'orange', linewidth=2, label='Confidence')
    ax2.plot(frames, thresholds, 'purple', linewidth=2, label='Adaptive Threshold')
    ax2.fill_between(frames, 0, confidences, alpha=0.3, color='orange')
    ax2.set_title('Confidence vs Adaptive Threshold', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Frame Index')
    ax2.set_ylabel('Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. Stability Analysis
    stabilities = [r['stabilized']['stability'] for r in results]
    stability_values = [1 if s == 'STABLE' else 0 for s in stabilities]
    
    if SCIPY_AVAILABLE:
        # Smooth the stability signal
        if len(stability_values) > 5:
            smoothed_stability = savgol_filter(stability_values, 
                                             min(len(stability_values), 5), 2)
        else:
            smoothed_stability = stability_values
        ax3.plot(frames, smoothed_stability, 'cyan', linewidth=3, label='Temporal Stability')
    else:
        ax3.plot(frames, stability_values, 'cyan', linewidth=3, label='Temporal Stability')
    
    ax3.fill_between(frames, 0, stability_values, alpha=0.3, color='cyan')
    ax3.set_title('Temporal Stability Analysis', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Frame Index')
    ax3.set_ylabel('Stability Score')
    ax3.set_ylim(-0.1, 1.1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Final Predictions
    final_preds = [1 if r['final']['prediction'] == 'REAL' else 0 for r in results]
    colors = ['green' if p == 1 else 'red' for p in final_preds]
    
    ax4.scatter(frames, final_preds, c=colors, s=100, alpha=0.7, edgecolors='black')
    ax4.set_title('Final Predictions', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Frame Index')
    ax4.set_ylabel('Prediction (0=FAKE, 1=REAL)')
    ax4.set_ylim(-0.1, 1.1)
    ax4.grid(True, alpha=0.3)
    
    # Add prediction counts as text
    real_count = sum(final_preds)
    fake_count = len(final_preds) - real_count
    ax4.text(0.02, 0.98, f'REAL: {real_count}\nFAKE: {fake_count}', 
             transform=ax4.transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    return fig

def calculate_video_statistics(results):
    """Calculate comprehensive statistics untuk video"""
    if not results:
        return {}
    
    # Extract predictions
    final_preds = [r['final']['prediction'] for r in results]
    confidences = [r['final']['confidence'] for r in results]
    stabilities = [r['stabilized']['stability'] for r in results]
    
    # Basic statistics
    real_count = sum(1 for p in final_preds if p == 'REAL')
    fake_count = len(final_preds) - real_count
    real_percentage = (real_count / len(final_preds)) * 100
    
    # Confidence statistics
    avg_confidence = np.mean(confidences)
    confidence_std = np.std(confidences)
    min_confidence = np.min(confidences)
    max_confidence = np.max(confidences)
    
    # Stability statistics
    stable_count = sum(1 for s in stabilities if s == 'STABLE')
    stability_percentage = (stable_count / len(stabilities)) * 100
    
    # Consistency analysis
    # Check for rapid changes in prediction
    changes = 0
    for i in range(1, len(final_preds)):
        if final_preds[i] != final_preds[i-1]:
            changes += 1
    
    consistency_score = 100 - (changes / (len(final_preds) - 1)) * 100 if len(final_preds) > 1 else 100
    
    # Enhanced metrics
    ensemble_weights = [(r['ensemble']['weights'][0], r['ensemble']['weights'][1]) for r in results]
    avg_mesonet_weight = np.mean([w[0] for w in ensemble_weights])
    avg_mobilevit_weight = np.mean([w[1] for w in ensemble_weights])
    
    return {
        'frame_count': len(results),
        'real_count': real_count,
        'fake_count': fake_count,
        'real_percentage': real_percentage,
        'fake_percentage': 100 - real_percentage,
        'avg_confidence': avg_confidence,
        'confidence_std': confidence_std,
        'min_confidence': min_confidence,
        'max_confidence': max_confidence,
        'stable_count': stable_count,
        'stability_percentage': stability_percentage,
        'consistency_score': consistency_score,
        'prediction_changes': changes,
        'avg_mesonet_weight': avg_mesonet_weight,
        'avg_mobilevit_weight': avg_mobilevit_weight
    }

# Main application
def main():
    st.markdown('<h1 class="main-header">🎬 Enhanced Video Deepfake Detection</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="algorithm-box">
    <h3>🚀 Enhanced MesoNet & MobileViT System</h3>
    <p>Sistem deteksi deepfake video yang dioptimalkan dengan algoritma enhancement:</p>
    <ul>
        <li><strong>Temporal Stabilization</strong> - Stabilisasi hasil antar frame</li>
        <li><strong>Adaptive Thresholding</strong> - Threshold yang menyesuaikan kondisi video</li>
        <li><strong>Dynamic Ensemble Optimization</strong> - Optimasi bobot MesoNet & MobileViT</li>
        <li><strong>Enhanced Normalization & Activation</strong> - BCN & AGLU yang ditingkatkan</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Enhanced Configuration")
    
    # Algorithm settings
    st.sidebar.subheader("🧠 Enhancement Algorithms")
    enable_temporal_stabilization = st.sidebar.checkbox("Temporal Stabilization", value=True)
    enable_adaptive_threshold = st.sidebar.checkbox("Adaptive Thresholding", value=True)
    enable_ensemble_optimization = st.sidebar.checkbox("Dynamic Ensemble Optimization", value=True)
    
    # Video processing settings
    st.sidebar.subheader("🎬 Video Processing")
    max_frames = st.sidebar.slider("Maximum Frames to Analyze", 10, 100, 30, 5)
    skip_frames = st.sidebar.slider("Frame Skip Interval", 1, 10, 5)
    use_face_detection = st.sidebar.checkbox("Face Detection", value=True)
    
    # Enhanced parameters
    st.sidebar.subheader("🔧 Advanced Parameters")
    stabilization_window = st.sidebar.slider("Stabilization Window", 3, 15, 7)
    stability_threshold = st.sidebar.slider("Stability Threshold", 0.1, 0.5, 0.25, 0.05)
    confidence_boost = st.sidebar.slider("Confidence Boost Factor", 0.9, 1.2, 1.1, 0.05)
    
    # Load models
    if not st.session_state.models_loaded:
        with st.spinner("🔄 Loading enhanced models and algorithms..."):
            models, detector = load_enhanced_models()
            st.session_state.models = models
            st.session_state.detector = detector
            st.session_state.models_loaded = True
    
    # System status
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔧 System Status")
    st.sidebar.write(f"TensorFlow: {'✅ Available' if TF_AVAILABLE else '❌ Not Available'}")
    st.sidebar.write(f"MTCNN: {'✅ Available' if MTCNN_AVAILABLE else '❌ Not Available'}")
    st.sidebar.write(f"SciPy: {'✅ Available' if SCIPY_AVAILABLE else '❌ Not Available'}")
    
    # Enhancement algorithms info
    with st.sidebar.expander("🧠 Enhancement Algorithms Info"):
        st.write("""
        **Temporal Stabilization**
        - Smooths predictions across frames
        - Reduces flickering effects
        - Uses weighted temporal averaging
        
        **Adaptive Thresholding**
        - Adjusts threshold based on video quality
        - Considers lighting conditions
        - Improves accuracy for varied content
        
        **Dynamic Ensemble Optimization**
        - Adapts MesoNet/MobileViT weights
        - Based on historical performance
        - Maximizes prediction accuracy
        
        **Enhanced BCN & AGLU**
        - Improved normalization techniques
        - Adaptive activation functions
        - Better feature representation
        """)
    
    # Main interface
    st.markdown('<h2 class="sub-header">🎬 Upload Video for Enhanced Analysis</h2>', 
                unsafe_allow_html=True)
    
    uploaded_video = st.file_uploader(
        "Choose a video file", 
        type=['mp4', 'avi', 'mov', 'mkv', 'wmv'],
        help="Upload a video file with clear faces for best results. Recommended: MP4 format, < 100MB"
    )
    
    if uploaded_video is not None:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📹 Video Information")
            
            # Display video info
            video_size = len(uploaded_video.getvalue())
            st.write(f"**File name:** {uploaded_video.name}")
            st.write(f"**File size:** {video_size / (1024*1024):.2f} MB")
            st.write(f"**File type:** {uploaded_video.type}")
            
            # Video preview (if possible)
            if video_size < 50 * 1024 * 1024:  # Less than 50MB
                st.video(uploaded_video)
            else:
                st.info("📹 Video too large for preview, but will be processed normally")
        
        with col2:
            st.subheader("⚙️ Processing Configuration")
            
            st.markdown(f"""
            <div class="video-stats">
            <strong>Processing Settings:</strong><br>
            • Max frames: {max_frames}<br>
            • Skip interval: {skip_frames}<br>
            • Face detection: {'Enabled' if use_face_detection else 'Disabled'}<br>
            • Temporal stabilization: {'Enabled' if enable_temporal_stabilization else 'Disabled'}<br>
            • Adaptive threshold: {'Enabled' if enable_adaptive_threshold else 'Disabled'}<br>
            • Ensemble optimization: {'Enabled' if enable_ensemble_optimization else 'Disabled'}
            </div>
            """, unsafe_allow_html=True)
        
        if st.button("🚀 Analyze Video with Enhanced Algorithms", type="primary"):
            
            # Enhanced processing workflow
            st.markdown("""
            <div class="enhancement-box">
            <h3>🔄 Enhanced Processing Pipeline</h3>
            <p>Starting comprehensive video analysis with enhanced algorithms...</p>
            </div>
            """, unsafe_allow_html=True)
            
            start_time = time.time()
            
            # Step 1: Extract frames
            st.subheader("📸 Frame Extraction")
            with st.spinner("Extracting frames from video..."):
                frames, frame_indices, total_frames, fps = extract_frames_from_video(
                    uploaded_video, max_frames, skip_frames
                )
            
            if not frames:
                st.error("❌ Failed to extract frames from video")
                return
            
            st.success(f"✅ Extracted {len(frames)} frames from {total_frames} total frames")
            
            # Step 2: Enhanced analysis
            st.subheader("🧠 Enhanced AI Analysis")
            with st.spinner("Running enhanced deepfake detection..."):
                results, face_frames = analyze_video_enhanced(
                    st.session_state.models, 
                    frames,
                    st.session_state.detector if use_face_detection else None
                )
            
            if not results:
                st.error("❌ Failed to analyze video frames")
                return
            
            processing_time = time.time() - start_time
            
            # Step 3: Results and visualization
            st.markdown("""
            <div class="stability-box">
            <h3>📊 Enhanced Analysis Results</h3>
            <p>Comprehensive video analysis completed with enhanced algorithms!</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Calculate statistics
            stats = calculate_video_statistics(results)
            
            # Main result summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if stats['real_percentage'] > 60:
                    st.metric("🎭 Final Verdict", "LIKELY REAL", f"{stats['real_percentage']:.1f}% frames")
                    verdict_color = "#28a745"
                elif stats['fake_percentage'] > 60:
                    st.metric("🚨 Final Verdict", "LIKELY FAKE", f"{stats['fake_percentage']:.1f}% frames")
                    verdict_color = "#dc3545"
                else:
                    st.metric("❓ Final Verdict", "UNCERTAIN", "Mixed signals")
                    verdict_color = "#ffc107"
            
            with col2:
                st.metric("📊 Avg Confidence", f"{stats['avg_confidence']:.3f}", 
                         f"±{stats['confidence_std']:.3f}")
            
            with col3:
                st.metric("🎯 Consistency", f"{stats['consistency_score']:.1f}%", 
                         f"{stats['prediction_changes']} changes")
            
            with col4:
                st.metric("⚡ Stability", f"{stats['stability_percentage']:.1f}%", 
                         f"{stats['stable_count']}/{stats['frame_count']}")
            
            # Enhanced statistics
            st.subheader("📈 Enhanced Analysis Dashboard")
            
            # Model weights analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**🤖 Dynamic Model Weights**")
                weight_data = pd.DataFrame({
                    'Model': ['MesoNet', 'MobileViT'],
                    'Average Weight': [stats['avg_mesonet_weight'], stats['avg_mobilevit_weight']],
                    'Contribution': [f"{stats['avg_mesonet_weight']*100:.1f}%", 
                                   f"{stats['avg_mobilevit_weight']*100:.1f}%"]
                })
                st.dataframe(weight_data, use_container_width=True)
            
            with col2:
                st.markdown("**📊 Performance Metrics**")
                perf_data = pd.DataFrame({
                    'Metric': ['Avg Confidence', 'Consistency Score', 'Stability Rate', 'Processing Time'],
                    'Value': [f"{stats['avg_confidence']:.3f}", 
                             f"{stats['consistency_score']:.1f}%",
                             f"{stats['stability_percentage']:.1f}%",
                             f"{processing_time:.2f}s"]
                })
                st.dataframe(perf_data, use_container_width=True)
            
            # Enhanced visualizations
            st.subheader("📊 Enhanced Temporal Analysis")
            fig = create_enhanced_visualizations(results)
            st.pyplot(fig)
            
            # Frame-by-frame analysis
            if st.checkbox("🔍 Show Frame-by-Frame Analysis"):
                st.subheader("🖼️ Detailed Frame Analysis")
                
                # Show sample frames with predictions
                num_samples = min(6, len(frames))
                sample_indices = np.linspace(0, len(frames)-1, num_samples, dtype=int)
                
                cols = st.columns(3)
                for i, idx in enumerate(sample_indices):
                    with cols[i % 3]:
                        st.image(face_frames[idx], caption=f"Frame {frame_indices[idx]}", 
                                use_column_width=True)
                        
                        result = results[idx]
                        pred_color = "green" if result['final']['prediction'] == 'REAL' else "red"
                        
                        st.markdown(f"""
                        <div style="background-color: #f8f9fa; padding: 0.5rem; border-radius: 5px; 
                                    border-left: 4px solid {pred_color};">
                        <strong>{result['final']['prediction']}</strong><br>
                        Confidence: {result['final']['confidence']:.3f}<br>
                        Stability: {result['stabilized']['stability']}<br>
                        MesoNet: {result['mesonet']['prob']:.3f}<br>
                        MobileViT: {result['mobilevit']['prob']:.3f}
                        </div>
                        """, unsafe_allow_html=True)
            
            # Detailed results table
            if st.checkbox("📋 Show Detailed Results Table"):
                st.subheader("📊 Complete Analysis Results")
                
                # Create detailed DataFrame
                detailed_results = []
                for i, result in enumerate(results):
                    detailed_results.append({
                        'Frame': frame_indices[i],
                        'Final_Prediction': result['final']['prediction'],
                        'Final_Confidence': f"{result['final']['confidence']:.3f}",
                        'MesoNet_Prob': f"{result['mesonet']['prob']:.3f}",
                        'MobileViT_Prob': f"{result['mobilevit']['prob']:.3f}",
                        'Ensemble_Prob': f"{result['ensemble']['prob']:.3f}",
                        'Stabilized_Prob': f"{result['stabilized']['prob']:.3f}",
                        'Stability': result['stabilized']['stability'],
                        'Adaptive_Threshold': f"{result['adaptive']['threshold']:.3f}",
                        'Frame_Quality': f"{result['adaptive']['quality']:.1f}",
                        'Lighting': f"{result['adaptive']['lighting']:.1f}"
                    })
                
                results_df = pd.DataFrame(detailed_results)
                st.dataframe(results_df, use_container_width=True)
                
                # Download option
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results as CSV",
                    data=csv,
                    file_name=f"video_analysis_{uploaded_video.name}_{int(time.time())}.csv",
                    mime="text/csv"
                )
            
            # Enhancement recommendations
            st.subheader("💡 Enhancement Recommendations")
            
            recommendations = []
            
            if stats['consistency_score'] < 70:
                recommendations.append("🔧 **Low consistency detected** - Consider increasing stabilization window")
            
            if stats['avg_confidence'] < 0.7:
                recommendations.append("⚠️ **Low average confidence** - Video quality may be poor or content is challenging")
            
            if stats['stability_percentage'] < 80:
                recommendations.append("📊 **Temporal instability** - Consider adjusting stability threshold")
            
            if abs(stats['avg_mesonet_weight'] - stats['avg_mobilevit_weight']) > 0.3:
                recommendations.append("⚖️ **Unbalanced model weights** - One model is dominating predictions")
            
            if stats['prediction_changes'] > len(results) * 0.3:
                recommendations.append("🎯 **High prediction variance** - Consider longer stabilization window")
            
            if not recommendations:
                recommendations.append("✅ **Excellent results** - All enhancement algorithms are working optimally")
            
            for rec in recommendations:
                st.markdown(f"- {rec}")
            
            # Final summary
            st.markdown(f"""
            <div class="algorithm-box">
            <h3>🎯 Enhanced Analysis Summary</h3>
            <p><strong>Video:</strong> {uploaded_video.name}</p>
            <p><strong>Frames Analyzed:</strong> {len(results)} of {total_frames} total frames</p>
            <p><strong>Processing Time:</strong> {processing_time:.2f} seconds</p>
            <p><strong>Enhanced Algorithms Used:</strong> Temporal Stabilization, Adaptive Thresholding, Dynamic Ensemble Optimization</p>
            <p><strong>Final Assessment:</strong> This video shows <strong>{'consistent' if stats['consistency_score'] > 80 else 'variable'}</strong> 
            patterns with <strong>{'high' if stats['avg_confidence'] > 0.8 else 'moderate' if stats['avg_confidence'] > 0.6 else 'low'}</strong> confidence.</p>
            </div>
            """, unsafe_allow_html=True)

    # Information section
    st.markdown("---")
    st.markdown('<h2 class="sub-header">🧠 Enhancement Algorithms Explained</h2>', 
                unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="enhancement-box">
        <h4>🎯 Temporal Stabilization</h4>
        <p><strong>Purpose:</strong> Mengurangi fluktuasi prediksi antar frame</p>
        <p><strong>Method:</strong> Weighted temporal averaging dengan decay factor</p>
        <p><strong>Benefit:</strong> Hasil yang lebih stabil dan konsisten</p>
        </div>
        
        <div class="algorithm-box">
        <h4>⚖️ Dynamic Ensemble Optimization</h4>
        <p><strong>Purpose:</strong> Mengoptimalkan bobot MesoNet dan MobileViT</p>
        <p><strong>Method:</strong> Adaptive weight adjustment berdasarkan performa</p>
        <p><strong>Benefit:</strong> Memaksimalkan akurasi kedua model</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stability-box">
        <h4>🔧 Adaptive Thresholding</h4>
        <p><strong>Purpose:</strong> Menyesuaikan threshold berdasarkan kondisi video</p>
        <p><strong>Method:</strong> Analisis kualitas frame dan lighting conditions</p>
        <p><strong>Benefit:</strong> Akurasi tinggi untuk berbagai jenis video</p>
        </div>
        
        <div class="enhancement-box">
        <h4>🚀 Enhanced BCN & AGLU</h4>
        <p><strong>Purpose:</strong> Meningkatkan representasi fitur pada model</p>
        <p><strong>Method:</strong> Improved normalization dan adaptive activation</p>
        <p><strong>Benefit:</strong> Feature extraction yang lebih baik</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()