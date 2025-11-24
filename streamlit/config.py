"""
Configuration file for Advanced Deepfake Detection System
========================================================
"""

# Model paths
MODEL_PATHS = {
    'mesonet': '/home/jak/myenv/skripsi_fix/cnn/model/trained_models/MesoNet_BCN_AGLU_final_model.h5',
    'mobilevit': '/home/jak/myenv/skripsi_fix/mobilevit/model/trained_models/MobileViT_XXS_Balanced_best.h5'
}

# Model configurations
MODEL_CONFIGS = {
    'mesonet': {
        'input_size': (224, 224),
        'name': 'MesoNet with BCN & AGLU',
        'description': 'Lightweight model specialized for deepfake detection',
        'weight': 1.2  # Higher weight due to specialization
    },
    'mobilevit': {
        'input_size': (256, 256),
        'name': 'MobileViT XXS',
        'description': 'Mobile-optimized Vision Transformer',
        'weight': 1.1
    },
    'efficientnet': {
        'input_size': (224, 224),
        'name': 'EfficientNet B0',
        'description': 'Efficient convolutional architecture',
        'weight': 1.0
    },
    'resnet50': {
        'input_size': (224, 224),
        'name': 'ResNet50',
        'description': 'Deep residual network',
        'weight': 0.9
    },
    'vit': {
        'input_size': (224, 224),
        'name': 'Vision Transformer',
        'description': 'Pure attention-based model',
        'weight': 1.0
    },
    'densenet': {
        'input_size': (224, 224),
        'name': 'DenseNet121',
        'description': 'Densely connected convolutional network',
        'weight': 0.95
    },
    'inception': {
        'input_size': (224, 224),
        'name': 'InceptionV3',
        'description': 'Multi-scale feature extraction',
        'weight': 0.9
    },
    'xception': {
        'input_size': (224, 224),
        'name': 'Xception',
        'description': 'Extreme inception architecture',
        'weight': 0.85
    }
}

# Detection thresholds
DETECTION_THRESHOLDS = {
    'high_precision': 0.7,
    'balanced': 0.5,
    'high_recall': 0.3
}

# MTCNN configuration
MTCNN_CONFIG = {
    'min_face_size': 20,
    'scale_factor': 0.709,
    'steps_threshold': [0.6, 0.7, 0.7],
    'confidence_threshold': 0.9
}

# Image preprocessing parameters
PREPROCESSING_CONFIG = {
    'normalization': 'standard',  # 'standard' or 'imagenet'
    'augmentation': {
        'rotation_range': 10,
        'width_shift_range': 0.1,
        'height_shift_range': 0.1,
        'horizontal_flip': True,
        'brightness_range': [0.9, 1.1]
    }
}

# Ensemble configuration
ENSEMBLE_CONFIG = {
    'voting_strategy': 'weighted',  # 'simple', 'weighted', 'confidence'
    'min_models': 3,  # Minimum models required for ensemble
    'consensus_threshold': 0.6  # Required agreement for high confidence
}

# UI configuration
UI_CONFIG = {
    'max_file_size': 10,  # MB
    'supported_formats': ['jpg', 'jpeg', 'png', 'bmp'],
    'default_models': ['mesonet', 'mobilevit', 'efficientnet'],
    'show_advanced_metrics': True
}

# Performance optimization
PERFORMANCE_CONFIG = {
    'batch_size': 1,
    'num_threads': 4,
    'gpu_memory_growth': True,
    'mixed_precision': False
}