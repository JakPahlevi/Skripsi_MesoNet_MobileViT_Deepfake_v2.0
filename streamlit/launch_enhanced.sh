#!/bin/bash

# Enhanced Video Deepfake Detection Launcher
# Implements Spatio-Temporal Consistency and Attention

echo "🎬 Enhanced Deepfake Detection with Spatio-Temporal Analysis"
echo "=========================================================="

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

# Check Python dependencies
echo "🧪 Checking dependencies..."
python -c "import streamlit, cv2, numpy, pandas, matplotlib" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️ Some dependencies missing. Installing..."
    pip install streamlit opencv-python numpy pandas matplotlib seaborn
fi

# Check TensorFlow
python -c "import tensorflow" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ TensorFlow available"
else
    echo "⚠️ TensorFlow not available - running in demo mode"
fi

# Check MTCNN
python -c "import mtcnn" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ MTCNN available"
else
    echo "⚠️ MTCNN not available - using fallback face detection"
fi

# Launch enhanced application
echo ""
echo "🚀 Launching Enhanced Deepfake Detection System..."
echo "📊 Features: Complete frame analysis, separate model results, temporal consistency"
echo ""
echo "🌐 Opening in browser..."

# Set environment variables for better performance
export TF_CPP_MIN_LOG_LEVEL=2
export OPENCV_LOG_LEVEL=ERROR

# Launch Streamlit
streamlit run enhanced_video_app.py \
    --server.port=8503 \
    --server.address=0.0.0.0 \
    --server.headless=false \
    --browser.gatherUsageStats=false \
    --server.fileWatcherType=none