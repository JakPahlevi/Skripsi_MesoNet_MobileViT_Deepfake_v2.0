"""
Utility functions for the Advanced Deepfake Detection System
===========================================================
"""

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import io
import base64
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

class ImageProcessor:
    """Image processing utilities"""
    
    @staticmethod
    def preprocess_for_model(image, target_size=(224, 224), model_type='standard'):
        """Preprocess image for specific model type"""
        try:
            # Convert PIL to numpy if needed
            if hasattr(image, 'mode'):
                image_array = np.array(image)
            else:
                image_array = image
            
            # Ensure RGB format
            if len(image_array.shape) == 3 and image_array.shape[2] == 3:
                # Resize
                resized = cv2.resize(image_array, target_size)
                
                # Normalize based on model type
                if model_type == 'imagenet':
                    # ImageNet normalization
                    resized = resized.astype(np.float32)
                    resized = (resized / 127.5) - 1.0  # [-1, 1] range
                else:
                    # Standard normalization
                    resized = resized.astype(np.float32) / 255.0  # [0, 1] range
                
                # Add batch dimension
                return np.expand_dims(resized, axis=0)
            else:
                return None
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    @staticmethod
    def enhance_image_quality(image):
        """Enhance image quality for better detection"""
        try:
            # Convert to numpy array if PIL
            if hasattr(image, 'mode'):
                img_array = np.array(image)
            else:
                img_array = image.copy()
            
            # Apply CLAHE for contrast enhancement
            if len(img_array.shape) == 3:
                # Convert to LAB color space
                lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
                l, a, b = cv2.split(lab)
                
                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l = clahe.apply(l)
                
                # Merge and convert back
                lab = cv2.merge((l, a, b))
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                # Grayscale
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(img_array)
            
            # Apply slight Gaussian blur to reduce noise
            enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
            
            return enhanced
            
        except Exception as e:
            print(f"Error enhancing image: {e}")
            return image

class ModelUtils:
    """Model-related utilities"""
    
    @staticmethod
    def load_model_safely(model_path, custom_objects=None):
        """Safely load a Keras model with error handling"""
        try:
            if custom_objects:
                model = tf.keras.models.load_model(model_path, custom_objects=custom_objects)
            else:
                model = tf.keras.models.load_model(model_path)
            return model, None
        except Exception as e:
            return None, str(e)
    
    @staticmethod
    def predict_with_error_handling(model, image, model_name):
        """Make prediction with comprehensive error handling"""
        try:
            if model is None:
                return None, 0.5, f"{model_name} not loaded"
            
            # Make prediction
            prediction = model.predict(image, verbose=0)
            
            # Handle different output formats
            if len(prediction.shape) > 1 and prediction.shape[1] == 2:
                # Binary classification with softmax
                prob_real = float(prediction[0][1])
                prob_fake = float(prediction[0][0])
            else:
                # Single output
                prob_real = float(prediction[0])
                prob_fake = 1 - prob_real
            
            # Determine result
            if prob_real > 0.5:
                result = "REAL"
                confidence = prob_real
            else:
                result = "FAKE"
                confidence = prob_fake
            
            return result, confidence, "Success"
            
        except Exception as e:
            return None, 0.5, f"Error: {str(e)}"

class EnsembleUtils:
    """Ensemble prediction utilities"""
    
    @staticmethod
    def weighted_voting(predictions, confidences, weights=None):
        """Weighted voting with confidence adjustment"""
        if not predictions or len(predictions) == 0:
            return "UNCERTAIN", 0.5
        
        if weights is None:
            weights = np.ones(len(predictions))
        
        # Normalize weights
        weights = np.array(weights) / np.sum(weights)
        
        # Calculate weighted scores
        fake_score = 0
        real_score = 0
        
        for i, (pred, conf) in enumerate(zip(predictions, confidences)):
            weighted_conf = weights[i] * conf
            if pred == "FAKE":
                fake_score += weighted_conf
            else:
                real_score += weighted_conf
        
        total_score = fake_score + real_score
        if total_score > 0:
            fake_score /= total_score
            real_score /= total_score
        
        if real_score > fake_score:
            return "REAL", real_score
        else:
            return "FAKE", fake_score
    
    @staticmethod
    def confidence_weighted_voting(predictions, confidences):
        """Voting weighted by prediction confidence"""
        if not predictions:
            return "UNCERTAIN", 0.5
        
        # Use confidence as weight
        return EnsembleUtils.weighted_voting(predictions, confidences, confidences)
    
    @staticmethod
    def consensus_analysis(predictions):
        """Analyze consensus among models"""
        if not predictions:
            return {"consensus": 0, "agreement": "No predictions"}
        
        fake_count = sum(1 for p in predictions if p == "FAKE")
        real_count = len(predictions) - fake_count
        
        consensus_ratio = max(fake_count, real_count) / len(predictions)
        
        if consensus_ratio >= 0.8:
            agreement = "Strong"
        elif consensus_ratio >= 0.6:
            agreement = "Moderate"
        else:
            agreement = "Weak"
        
        return {
            "consensus": consensus_ratio,
            "agreement": agreement,
            "fake_votes": fake_count,
            "real_votes": real_count,
            "total_models": len(predictions)
        }

class VisualizationUtils:
    """Visualization utilities"""
    
    @staticmethod
    def create_confidence_chart(model_names, confidences, predictions):
        """Create confidence visualization chart"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['red' if pred == 'FAKE' else 'green' for pred in predictions]
        bars = ax.bar(model_names, confidences, color=colors, alpha=0.7)
        
        ax.set_ylabel('Confidence Score')
        ax.set_title('Model Confidence Scores')
        ax.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, conf in zip(bars, confidences):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{conf:.3f}', ha='center', va='bottom')
        
        # Add legend
        fake_patch = plt.Rectangle((0, 0), 1, 1, color='red', alpha=0.7, label='FAKE')
        real_patch = plt.Rectangle((0, 0), 1, 1, color='green', alpha=0.7, label='REAL')
        ax.legend(handles=[fake_patch, real_patch])
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig
    
    @staticmethod
    def create_consensus_pie_chart(consensus_data):
        """Create consensus analysis pie chart"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        labels = ['FAKE votes', 'REAL votes']
        sizes = [consensus_data['fake_votes'], consensus_data['real_votes']]
        colors = ['red', 'green']
        explode = (0.05, 0.05)
        
        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, 
                                         colors=colors, autopct='%1.1f%%',
                                         shadow=True, startangle=90)
        
        ax.set_title(f'Model Consensus Analysis\n{consensus_data["agreement"]} Agreement')
        
        return fig
    
    @staticmethod
    def plot_to_base64(fig):
        """Convert matplotlib figure to base64 string"""
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png', bbox_inches='tight', dpi=150)
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        buffer.close()
        plt.close(fig)
        return image_base64

class ValidationUtils:
    """Input validation utilities"""
    
    @staticmethod
    def validate_image(uploaded_file, max_size_mb=10):
        """Validate uploaded image file"""
        if uploaded_file is None:
            return False, "No file uploaded"
        
        # Check file size
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > max_size_mb:
            return False, f"File size ({file_size_mb:.1f}MB) exceeds limit ({max_size_mb}MB)"
        
        # Check file type
        allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/bmp']
        if uploaded_file.type not in allowed_types:
            return False, f"Unsupported file type: {uploaded_file.type}"
        
        try:
            # Try to load image
            image = Image.open(uploaded_file)
            
            # Check image dimensions
            width, height = image.size
            if width < 64 or height < 64:
                return False, "Image too small (minimum 64x64 pixels)"
            
            if width > 4096 or height > 4096:
                return False, "Image too large (maximum 4096x4096 pixels)"
            
            return True, "Valid image"
            
        except Exception as e:
            return False, f"Invalid image file: {str(e)}"
    
    @staticmethod
    def validate_model_selection(selected_models):
        """Validate model selection"""
        if not selected_models:
            return False, "No models selected"
        
        if len(selected_models) < 2:
            return False, "Select at least 2 models for reliable detection"
        
        return True, "Valid model selection"

class PerformanceUtils:
    """Performance monitoring utilities"""
    
    @staticmethod
    def measure_inference_time(func, *args, **kwargs):
        """Measure function execution time"""
        import time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time
    
    @staticmethod
    def get_model_memory_usage(model):
        """Estimate model memory usage"""
        try:
            total_params = model.count_params()
            # Assuming float32 (4 bytes per parameter)
            memory_mb = (total_params * 4) / (1024 * 1024)
            return memory_mb
        except:
            return 0
    
    @staticmethod
    def optimize_tensorflow():
        """Optimize TensorFlow settings for inference"""
        try:
            # Enable GPU memory growth if available
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set CPU thread count
            tf.config.threading.set_inter_op_parallelism_threads(4)
            tf.config.threading.set_intra_op_parallelism_threads(4)
            
        except Exception as e:
            print(f"TensorFlow optimization warning: {e}")

class ReportGenerator:
    """Generate detailed analysis reports"""
    
    @staticmethod
    def generate_detection_report(results, image_info, processing_time):
        """Generate comprehensive detection report"""
        report = {
            "timestamp": str(np.datetime64('now')),
            "image_info": image_info,
            "processing_time": processing_time,
            "models_used": len(results),
            "results": results,
            "consensus": EnsembleUtils.consensus_analysis([r[1] for r in results]),
            "confidence_stats": {
                "mean": np.mean([float(r[2]) for r in results]),
                "std": np.std([float(r[2]) for r in results]),
                "min": np.min([float(r[2]) for r in results]),
                "max": np.max([float(r[2]) for r in results])
            }
        }
        
        return report
    
    @staticmethod
    def format_report_for_display(report):
        """Format report for Streamlit display"""
        formatted = f"""
        ## 📋 Detection Report
        
        **Analysis Time:** {report['timestamp']}  
        **Processing Duration:** {report['processing_time']:.2f} seconds  
        **Models Used:** {report['models_used']}  
        
        ### 📊 Consensus Analysis
        - **Agreement Level:** {report['consensus']['agreement']}
        - **Consensus Ratio:** {report['consensus']['consensus']:.2%}
        - **FAKE Votes:** {report['consensus']['fake_votes']}
        - **REAL Votes:** {report['consensus']['real_votes']}
        
        ### 📈 Confidence Statistics
        - **Mean Confidence:** {report['confidence_stats']['mean']:.4f}
        - **Std Deviation:** {report['confidence_stats']['std']:.4f}
        - **Range:** {report['confidence_stats']['min']:.4f} - {report['confidence_stats']['max']:.4f}
        """
        
        return formatted