#!/bin/bash

# Advanced Deepfake Detection System Launcher
# ===========================================

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
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

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to activate virtual environment
activate_venv() {
    if [ -d "venv" ]; then
        print_status "Activating virtual environment..."
        source venv/bin/activate
        print_success "Virtual environment activated"
        return 0
    else
        print_warning "Virtual environment not found. Run setup first."
        return 1
    fi
}

# Function to check dependencies
check_dependencies() {
    print_status "Checking dependencies..."
    
    if ! activate_venv; then
        return 1
    fi
    
    # Check Python packages
    local missing_packages=()
    
    if ! python -c "import streamlit" 2>/dev/null; then
        missing_packages+=("streamlit")
    fi
    
    if ! python -c "import tensorflow" 2>/dev/null; then
        missing_packages+=("tensorflow")
    fi
    
    if ! python -c "import cv2" 2>/dev/null; then
        missing_packages+=("opencv-python")
    fi
    
    if ! python -c "import numpy" 2>/dev/null; then
        missing_packages+=("numpy")
    fi
    
    if [ ${#missing_packages[@]} -eq 0 ]; then
        print_success "All required dependencies are available"
        return 0
    else
        print_error "Missing dependencies: ${missing_packages[*]}"
        print_status "Run './setup.sh' to install dependencies"
        return 1
    fi
}

# Function to run video app
run_video_app() {
    print_status "Starting Enhanced Video Deepfake Detection App..."
    print_status "This version specializes in video processing with enhancement algorithms"
    echo
    
    if activate_venv; then
        if [ -f "video_app_fixed.py" ]; then
            print_status "Launching fixed video application..."
            echo
            print_success "🎬 Video Detection App will open in your browser at http://localhost:8501"
            print_success "📹 Upload your video file for enhanced temporal analysis!"
            echo
            streamlit run video_app_fixed.py --server.maxUploadSize 200
        elif [ -f "video_app.py" ]; then
            print_warning "Using original video app (may have compatibility issues)"
            streamlit run video_app.py --server.maxUploadSize 200
        else
            print_error "video_app_fixed.py not found in current directory"
            return 1
        fi
    else
        return 1
    fi
}

# Function to run simple app
run_simple_app() {
    print_status "Starting Simple Deepfake Detection App..."
    print_status "This version includes fallback mechanisms for missing dependencies"
    echo
    
    if activate_venv; then
        if [ -f "simple_app.py" ]; then
            print_status "Launching application..."
            echo
            print_success "🚀 Application will open in your browser at http://localhost:8501"
            echo
            streamlit run simple_app.py
        else
            print_error "simple_app.py not found in current directory"
            return 1
        fi
    else
        return 1
    fi
}

# Function to run advanced app
run_advanced_app() {
    print_status "Starting Advanced Deepfake Detection App..."
    print_status "This version requires all dependencies to be installed"
    echo
    
    if ! check_dependencies; then
        print_error "Dependencies check failed. Cannot run advanced app."
        print_status "Try running the simple app instead: ./launch.sh simple"
        return 1
    fi
    
    if [ -f "app.py" ]; then
        print_status "Launching advanced application..."
        echo
        print_success "🚀 Application will open in your browser at http://localhost:8501"
        echo
        streamlit run app.py
    else
        print_error "app.py not found in current directory"
        return 1
    fi
}

# Function to run setup
run_setup() {
    print_status "Running setup..."
    if [ -f "setup.sh" ]; then
        chmod +x setup.sh
        ./setup.sh
    else
        print_error "setup.sh not found"
        return 1
    fi
}

# Function to show status
show_status() {
    echo
    echo "==============================================="
    echo "🤖 Advanced Deepfake Detection System Status"
    echo "==============================================="
    echo
    
    # Check Python
    if command_exists python3; then
        print_success "Python 3: $(python3 --version)"
    else
        print_error "Python 3: Not found"
    fi
    
    # Check pip
    if command_exists pip3; then
        print_success "pip3: $(pip3 --version | cut -d' ' -f1-2)"
    else
        print_error "pip3: Not found"
    fi
    
    # Check virtual environment
    if [ -d "venv" ]; then
        print_success "Virtual Environment: Available"
        
        # Check if activated
        if [[ "$VIRTUAL_ENV" != "" ]]; then
            print_success "Virtual Environment: Activated ($VIRTUAL_ENV)"
        else
            print_warning "Virtual Environment: Not activated"
        fi
    else
        print_warning "Virtual Environment: Not found (run setup)"
    fi
    
    # Check files
    echo
    print_status "Application Files:"
    
    if [ -f "simple_app.py" ]; then
        print_success "simple_app.py: Available"
    else
        print_error "simple_app.py: Missing"
    fi
    
    if [ -f "video_app_fixed.py" ]; then
        print_success "video_app_fixed.py: Available (Enhanced Video Detection)"
    elif [ -f "video_app.py" ]; then
        print_warning "video_app.py: Available (Original, may have issues)"
    else
        print_warning "video_app_fixed.py: Missing"
    fi
    
    if [ -f "requirements.txt" ]; then
        print_success "requirements.txt: Available"
    else
        print_warning "requirements.txt: Missing"
    fi
    
    # Check dependencies if venv exists
    if [ -d "venv" ]; then
        echo
        print_status "Checking Python Dependencies..."
        
        if activate_venv; then
            local deps=("streamlit" "tensorflow" "opencv-python" "numpy" "pandas" "matplotlib")
            
            for dep in "${deps[@]}"; do
                if python -c "import ${dep//-/_}" 2>/dev/null; then
                    print_success "$dep: Available"
                else
                    print_warning "$dep: Missing"
                fi
            done
            
            # Check optional dependencies
            echo
            print_status "Optional Dependencies:"
            
            if python -c "import mtcnn" 2>/dev/null; then
                print_success "mtcnn: Available (face detection enabled)"
            else
                print_warning "mtcnn: Missing (fallback face detection will be used)"
            fi
            
            if python -c "import sklearn" 2>/dev/null; then
                print_success "scikit-learn: Available (ensemble methods enabled)"
            else
                print_warning "scikit-learn: Missing (basic ensemble only)"
            fi
        fi
    fi
    
    echo
    print_status "System Ready: Use './launch.sh video' for video processing or './launch.sh simple' for images"
    echo
}

# Function to show help
show_help() {
    echo
    echo "🤖 Advanced Deepfake Detection System Launcher"
    echo "=============================================="
    echo
    echo "Usage: ./launch.sh [command]"
    echo
    echo "Commands:"
    echo "  simple     - Run simple app (recommended, with fallbacks)"
    echo "  video      - Run enhanced video detection app (NEW!)"
    echo "  advanced   - Run advanced app (requires all dependencies)"
    echo "  setup      - Run setup script to install dependencies"
    echo "  status     - Show system status and dependencies"
    echo "  help       - Show this help message"
    echo
    echo "Examples:"
    echo "  ./launch.sh video      # Start video detection app (RECOMMENDED)"
    echo "  ./launch.sh simple     # Start simple image application"
    echo "  ./launch.sh setup      # Install dependencies"
    echo "  ./launch.sh status     # Check system status"
    echo
    echo "Quick Start:"
    echo "  1. ./launch.sh setup    # First time setup"
    echo "  2. ./launch.sh video    # Run video detection app"
    echo
    echo "Troubleshooting:"
    echo "  - If setup fails, check internet connection and Python installation"
    echo "  - If app won't start, try './launch.sh status' to check dependencies"
    echo "  - For port conflicts, use: streamlit run simple_app.py --server.port 8502"
    echo
}

# Main script logic
case "${1:-}" in
    "simple")
        run_simple_app
        ;;
    "advanced")
        run_advanced_app
        ;;
    "video")
        run_video_app
        ;;
    "setup")
        run_setup
        ;;
    "status")
        show_status
        ;;
    "help"|"-h"|"--help")
        show_help
        ;;
    "")
        print_status "No command specified. Showing status..."
        show_status
        echo
        print_status "Use './launch.sh help' for usage information"
        ;;
    *)
        print_error "Unknown command: $1"
        print_status "Use './launch.sh help' for usage information"
        exit 1
        ;;
esac