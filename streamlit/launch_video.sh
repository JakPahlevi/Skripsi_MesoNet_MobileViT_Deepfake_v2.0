#!/bin/bash

# Enhanced Video Deepfake Detection Launcher
# ==========================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_enhancement() {
    echo -e "${PURPLE}[ENHANCEMENT]${NC} $1"
}

# Header
echo
echo "=========================================================="
echo "🎬 Enhanced Video Deepfake Detection System"
echo "=========================================================="
echo "🚀 Optimized MesoNet & MobileViT with Enhancement Algorithms"
echo "📹 Specialized for Video Processing with Temporal Stability"
echo "=========================================================="
echo

# Check if virtual environment exists
if [ -d "venv" ]; then
    print_status "Activating virtual environment..."
    source venv/bin/activate
    print_success "Virtual environment activated"
else
    print_warning "Virtual environment not found. Run setup first."
    print_status "Would you like to run setup now? (y/n)"
    read -r response
    if [[ "$response" =~ ^[Yy]$ ]]; then
        if [ -f "setup.sh" ]; then
            ./setup.sh
            source venv/bin/activate
        else
            print_error "setup.sh not found"
            exit 1
        fi
    else
        print_error "Cannot proceed without virtual environment"
        exit 1
    fi
fi

# Check for video processing dependencies
print_status "Checking video processing capabilities..."

# Check OpenCV
if python -c "import cv2; print(f'OpenCV {cv2.__version__} ready for video processing')" 2>/dev/null; then
    print_success "OpenCV available for video processing"
else
    print_error "OpenCV not available - required for video processing"
    print_status "Installing OpenCV..."
    pip install opencv-python
fi

# Check additional video dependencies
print_status "Checking additional dependencies..."

if python -c "import scipy" 2>/dev/null; then
    print_success "SciPy available for enhanced algorithms"
else
    print_warning "SciPy not available - some enhancement algorithms will be limited"
    print_status "Installing SciPy for enhanced algorithms..."
    pip install scipy
fi

# Display enhancement features
echo
print_enhancement "🧠 Enhancement Algorithms Loaded:"
print_enhancement "  ✅ Temporal Stabilization - Reduces frame-to-frame flickering"
print_enhancement "  ✅ Adaptive Thresholding - Adjusts to video quality conditions"
print_enhancement "  ✅ Dynamic Ensemble Optimization - Maximizes MesoNet & MobileViT performance"
print_enhancement "  ✅ Enhanced BCN & AGLU - Improved normalization and activation"
echo

# Video processing information
print_status "📹 Video Processing Features:"
echo "   • Supports MP4, AVI, MOV, MKV, WMV formats"
echo "   • Automatic frame extraction and sampling"
echo "   • MTCNN face detection for optimal accuracy"
echo "   • Real-time processing with progress tracking"
echo "   • Comprehensive temporal analysis and visualization"
echo

# Check if video app exists
if [ -f "video_app_fixed.py" ]; then
    print_success "Enhanced Video Detection App found (Fixed Version)"
    
    # Display recommended settings
    print_status "💡 Recommended Settings for Best Results:"
    echo "   • Max Frames: 30-50 for detailed analysis"
    echo "   • Skip Frames: 3-5 for good temporal coverage"
    echo "   • Enable all enhancement algorithms"
    echo "   • Use face detection for better accuracy"
    echo

    print_status "🚀 Launching Enhanced Video Deepfake Detection System..."
    echo
    print_success "🎬 Application will open in your browser at http://localhost:8501"
    print_success "📹 Upload your video file and experience enhanced detection!"
    echo
    
    # Launch the application
    streamlit run video_app_fixed.py --server.maxUploadSize 200
    
elif [ -f "video_app.py" ]; then
    print_success "Enhanced Video Detection App found"
    print_warning "Using original version - may have some compatibility issues"
    streamlit run video_app.py --server.maxUploadSize 200
else
    print_error "video_app_fixed.py not found in current directory"
    print_status "Available files:"
    ls -la *.py 2>/dev/null || echo "No Python files found"
    exit 1
fi