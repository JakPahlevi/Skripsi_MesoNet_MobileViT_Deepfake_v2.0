"""
Enhanced Model Utilities for Deepfake Detection
==============================================

Additional models and ensemble methods to improve detection accuracy
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import DenseNet121, InceptionV3, Xception
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import cv2

class EnhancedMesoNet:
    """Enhanced MesoNet with multiple variants"""
    
    @staticmethod
    def create_meso_inception(input_shape=(224, 224, 3), num_classes=2):
        """MesoNet with Inception-like blocks"""
        inputs = keras.Input(shape=input_shape)
        
        # Inception block 1
        branch1x1 = layers.Conv2D(8, (1, 1), padding='same', activation='relu')(inputs)
        
        branch3x3 = layers.Conv2D(8, (1, 1), padding='same', activation='relu')(inputs)
        branch3x3 = layers.Conv2D(8, (3, 3), padding='same', activation='relu')(branch3x3)
        
        branch5x5 = layers.Conv2D(8, (1, 1), padding='same', activation='relu')(inputs)
        branch5x5 = layers.Conv2D(8, (5, 5), padding='same', activation='relu')(branch5x5)
        
        branch_pool = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
        branch_pool = layers.Conv2D(8, (1, 1), padding='same', activation='relu')(branch_pool)
        
        x = layers.concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=3)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Inception block 2
        branch1x1 = layers.Conv2D(16, (1, 1), padding='same', activation='relu')(x)
        
        branch3x3 = layers.Conv2D(16, (1, 1), padding='same', activation='relu')(x)
        branch3x3 = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(branch3x3)
        
        branch5x5 = layers.Conv2D(16, (1, 1), padding='same', activation='relu')(x)
        branch5x5 = layers.Conv2D(16, (5, 5), padding='same', activation='relu')(branch5x5)
        
        branch_pool = layers.MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
        branch_pool = layers.Conv2D(16, (1, 1), padding='same', activation='relu')(branch_pool)
        
        x = layers.concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=3)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Classification head
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='MesoInception')
        return model
    
    @staticmethod
    def create_meso_attention(input_shape=(224, 224, 3), num_classes=2):
        """MesoNet with attention mechanism"""
        inputs = keras.Input(shape=input_shape)
        
        # Feature extraction
        x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.MaxPooling2D((2, 2))(x)
        
        # Attention mechanism
        attention = layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x)
        x = layers.multiply([x, attention])
        
        # Classification
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='MesoAttention')
        return model

class AdditionalModels:
    """Additional state-of-the-art models for ensemble"""
    
    @staticmethod
    def create_densenet_model(input_shape=(224, 224, 3), num_classes=2):
        """DenseNet121 for deepfake detection"""
        base_model = DenseNet121(
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
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='DenseNet121_Deepfake')
        return model
    
    @staticmethod
    def create_inception_model(input_shape=(224, 224, 3), num_classes=2):
        """InceptionV3 for deepfake detection"""
        base_model = InceptionV3(
            weights='imagenet',
            include_top=False,
            input_shape=input_shape
        )
        
        # Freeze early layers
        for layer in base_model.layers[:-40]:
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
        
        model = Model(inputs, outputs, name='InceptionV3_Deepfake')
        return model
    
    @staticmethod
    def create_xception_model(input_shape=(224, 224, 3), num_classes=2):
        """Xception for deepfake detection"""
        base_model = Xception(
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
        x = layers.Dense(512, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs, outputs, name='Xception_Deepfake')
        return model

class AdvancedEnsemble:
    """Advanced ensemble methods for improved accuracy"""
    
    @staticmethod
    def create_stacking_ensemble():
        """Create stacking ensemble with multiple base learners"""
        base_learners = [
            ('ada', AdaBoostClassifier(n_estimators=100, random_state=42)),
            ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
            ('mlp', MLPClassifier(hidden_layer_sizes=(256, 128), random_state=42, max_iter=500))
        ]
        return base_learners
    
    @staticmethod
    def weighted_ensemble_prediction(predictions, weights=None):
        """Weighted ensemble prediction with confidence weighting"""
        if weights is None:
            weights = np.ones(len(predictions)) / len(predictions)
        
        weighted_sum = np.zeros(2)  # Assuming binary classification
        total_weight = 0
        
        for i, (pred, confidence) in enumerate(predictions):
            weight = weights[i] * confidence
            if pred == "FAKE":
                weighted_sum[0] += weight
            else:
                weighted_sum[1] += weight
            total_weight += weight
        
        if total_weight > 0:
            weighted_sum /= total_weight
        
        if weighted_sum[1] > weighted_sum[0]:
            return "REAL", weighted_sum[1]
        else:
            return "FAKE", weighted_sum[0]

class ImageEnhancement:
    """Image enhancement techniques for better detection"""
    
    @staticmethod
    def enhance_image_contrast(image):
        """Enhance image contrast using CLAHE"""
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back to RGB
            lab = cv2.merge((l, a, b))
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        
        return enhanced
    
    @staticmethod
    def apply_gaussian_filter(image, kernel_size=5):
        """Apply Gaussian filter for noise reduction"""
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    @staticmethod
    def edge_enhancement(image):
        """Enhance edges using unsharp masking"""
        # Create Gaussian blur
        gaussian = cv2.GaussianBlur(image, (0, 0), 2.0)
        
        # Create unsharp mask
        unsharp_mask = cv2.addWeighted(image, 1.5, gaussian, -0.5, 0)
        
        return unsharp_mask

class MetricsAnalyzer:
    """Advanced metrics and analysis for model evaluation"""
    
    @staticmethod
    def calculate_ensemble_metrics(predictions, confidences, ground_truth=None):
        """Calculate comprehensive ensemble metrics"""
        metrics = {
            'total_models': len(predictions),
            'fake_predictions': sum(1 for p in predictions if p == 'FAKE'),
            'real_predictions': sum(1 for p in predictions if p == 'REAL'),
            'average_confidence': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'confidence_range': np.max(confidences) - np.min(confidences),
            'consensus_strength': max(
                sum(1 for p in predictions if p == 'FAKE'),
                sum(1 for p in predictions if p == 'REAL')
            ) / len(predictions)
        }
        
        # Calculate uncertainty
        fake_conf = np.mean([conf for i, conf in enumerate(confidences) if predictions[i] == 'FAKE'])
        real_conf = np.mean([conf for i, conf in enumerate(confidences) if predictions[i] == 'REAL'])
        
        if not np.isnan(fake_conf) and not np.isnan(real_conf):
            metrics['uncertainty'] = abs(fake_conf - real_conf)
        else:
            metrics['uncertainty'] = 0
        
        return metrics
    
    @staticmethod
    def generate_confidence_intervals(confidences, alpha=0.05):
        """Generate confidence intervals for predictions"""
        n = len(confidences)
        mean_conf = np.mean(confidences)
        std_conf = np.std(confidences)
        
        # Calculate t-statistic (assuming normal distribution)
        from scipy import stats
        t_stat = stats.t.ppf(1 - alpha/2, n-1)
        margin_error = t_stat * (std_conf / np.sqrt(n))
        
        return {
            'mean': mean_conf,
            'lower_bound': mean_conf - margin_error,
            'upper_bound': mean_conf + margin_error,
            'margin_error': margin_error
        }