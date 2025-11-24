#!/bin/bash

# Enhanced Video Deepfake Detection Launcher - FIXED VERSION
# Implements Spatio-Temporal Consistency and Attention with proper dependencies

echo "🎬 Enhanced Deepfake Detection with Spatio-Temporal Analysis (FIXED)"
echo "=================================================================="

# Check if we're in the right directory
if [ ! -f "enhanced_video_app.py" ]; then
    echo "❌ Enhanced video app not found. Please run from streamlit directory."
    exit 1
fi

# Check virtual environment
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found. Please run setup.sh first."
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Fix numpy version for TensorFlow compatibility
echo "🔧 Ensuring TensorFlow compatibility..."
pip install numpy==1.26.4 --force-reinstall --quiet

# Check dependencies with better error handling
echo "🧪 Checking dependencies..."

# Check TensorFlow
python -c "import tensorflow as tf; print('✅ TensorFlow', tf.__version__, 'available')" 2>/dev/null
if [ $? -eq 0 ]; then
    TF_STATUS="✅ Available"
else
    echo "❌ TensorFlow not available"
    TF_STATUS="❌ Not Available"
fi

# Check MTCNN
python -c "from mtcnn import MTCNN; print('✅ MTCNN available')" 2>/dev/null
if [ $? -eq 0 ]; then
    MTCNN_STATUS="✅ Available"
else
    echo "❌ MTCNN not available"
    MTCNN_STATUS="❌ Not Available"
fi

# Check OpenCV
python -c "import cv2; print('✅ OpenCV', cv2.__version__, 'available')" 2>/dev/null
if [ $? -eq 0 ]; then
    CV_STATUS="✅ Available"
else
    echo "❌ OpenCV not available"
    CV_STATUS="❌ Not Available"
fi

# Display status
echo ""
echo "📋 System Status:"
echo "  TensorFlow: $TF_STATUS"
echo "  MTCNN: $MTCNN_STATUS"
echo "  OpenCV: $CV_STATUS"
echo ""

# Launch enhanced application
echo "🚀 Launching Enhanced Deepfake Detection System..."
echo "📊 Features:"
echo "  - ✅ Complete frame analysis (no 30-frame limit)"
echo "  - ✅ Separate MesoNet & MobileViT results"
echo "  - ✅ Spatio-Temporal Consistency & Attention"
echo "  - ✅ Comprehensive result tables"
echo "  - ✅ Advanced visualizations"
echo ""
echo "🌐 Opening in browser..."

# Set environment variables for better performance and reduced warnings
export TF_CPP_MIN_LOG_LEVEL=2
export OPENCV_LOG_LEVEL=ERROR
export TF_ENABLE_ONEDNN_OPTS=0

# Launch Streamlit with enhanced settings
streamlit run enhanced_video_app.py \
    --server.port=8504 \
    --server.address=0.0.0.0 \
    --server.headless=false \
    --browser.gatherUsageStats=false \
    --server.fileWatcherType=none \
    --server.maxUploadSize=500