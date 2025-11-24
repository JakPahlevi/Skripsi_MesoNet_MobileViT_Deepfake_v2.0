#!/bin/bash

# Advanced Deepfake Detection System Setup Script
# ===============================================

echo "🚀 Setting up Advanced Deepfake Detection System..."
echo "=================================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or later."
    exit 1
fi

echo "✅ Python 3 found: $(python3 --version)"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is not installed. Please install pip3."
    exit 1
fi

echo "✅ pip3 found: $(pip3 --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "🔧 Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created."
else
    echo "✅ Virtual environment already exists."
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "🔧 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing requirements..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo "✅ Requirements installed successfully."
else
    echo "❌ requirements.txt not found. Installing basic packages..."
    pip install streamlit tensorflow opencv-python numpy pandas matplotlib seaborn scikit-learn pillow mtcnn plotly xgboost
fi

# Check if TensorFlow can import
echo "🧪 Testing TensorFlow installation..."
python3 -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')" && echo "✅ TensorFlow working correctly." || echo "⚠️ TensorFlow installation issue detected."

# Check if OpenCV can import
echo "🧪 Testing OpenCV installation..."
python3 -c "import cv2; print(f'OpenCV version: {cv2.__version__}')" && echo "✅ OpenCV working correctly." || echo "⚠️ OpenCV installation issue detected."

# Check if MTCNN can import
echo "🧪 Testing MTCNN installation..."
python3 -c "from mtcnn import MTCNN; print('MTCNN imported successfully')" && echo "✅ MTCNN working correctly." || echo "⚠️ MTCNN installation issue detected."

echo ""
echo "🎉 Setup completed!"
echo "=================================================="
echo ""
echo "📖 Quick Start Guide:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run the simple app: streamlit run simple_app.py"
echo "3. Run the advanced app: streamlit run app.py"
echo ""
echo "🌐 The application will open in your web browser automatically."
echo ""
echo "📝 Note: If any tests failed above, you may need to:"
echo "   - Install system dependencies (e.g., libgl1-mesa-glx for OpenCV)"
echo "   - Update your system packages"
echo "   - Try installing packages individually if there are conflicts"
echo ""
echo "💡 For help, check the README.md file or run: streamlit run simple_app.py --help"