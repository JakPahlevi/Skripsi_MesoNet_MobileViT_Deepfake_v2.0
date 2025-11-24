# 🤖 Advanced Deepfake Detection System

**Sistem Deteksi Deepfake Multi-Model dengan Ensemble Voting**

Sistem canggih untuk deteksi deepfake yang menggabungkan multiple state-of-the-art models untuk meningkatkan akurasi deteksi. Dirancang khusus untuk mengatasi masalah akurasi yang rendah pada model tunggal MesoNet dan MobileViT.

## 🌟 Fitur Utama

### 🎯 Multi-Model Architecture
- **MesoNet dengan BCN & AGLU** - Model khusus deepfake dengan enhancement
- **MobileViT XXS** - Vision Transformer yang dioptimasi untuk mobile
- **EfficientNet B0** - Arsitektur convolutional yang efisien
- **ResNet50** - Deep residual network
- **Vision Transformer (ViT)** - Pure attention-based model
- **DenseNet121** - Densely connected network
- **InceptionV3** - Multi-scale feature extraction

### 🔍 Advanced Detection Features
- **MTCNN Face Detection** - Deteksi wajah otomatis dengan confidence scoring
- **Ensemble Voting** - Kombinasi prediksi dari multiple models
- **Weighted Predictions** - Bobot berbeda untuk setiap model berdasarkan performa
- **Confidence Analysis** - Analisis tingkat kepercayaan prediksi
- **Real-time Processing** - Interface responsif dengan feedback real-time

### 📊 Comprehensive Analysis
- **Interactive Dashboard** - Visualisasi hasil dengan Plotly dan Matplotlib
- **Statistical Analysis** - Analisis statistik mendalam dari hasil prediksi
- **Model Performance Comparison** - Perbandingan performa antar model
- **Consensus Analysis** - Analisis kesepakatan antar model
- **Detailed Reporting** - Laporan lengkap dengan rekomendasi

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone atau download project
cd /home/jak/myenv/skripsi_fix/streamlit

# Jalankan setup script
chmod +x setup.sh
./setup.sh
```

### 2. Jalankan Aplikasi

#### Simple App (Recommended untuk testing)
```bash
# Aktivasi virtual environment
source venv/bin/activate

# Jalankan aplikasi sederhana
streamlit run simple_app.py
```

#### Advanced App (Full features)
```bash
# Aktivasi virtual environment
source venv/bin/activate

# Jalankan aplikasi lengkap
streamlit run app.py
```

### 3. Akses Web Interface
- Aplikasi akan terbuka otomatis di browser
- Default URL: `http://localhost:8501`

## 📁 Struktur Project

```
streamlit/
├── simple_app.py              # Aplikasi sederhana dengan fallback
├── app.py                     # Aplikasi lengkap dengan semua fitur
├── enhanced_models.py         # Model tambahan dan ensemble methods
├── config.py                  # Konfigurasi sistem
├── utils.py                   # Utility functions
├── requirements.txt           # Dependencies
├── setup.sh                   # Setup script
├── README.md                  # Dokumentasi ini
└── venv/                      # Virtual environment (dibuat otomatis)
```

## 🔧 Manual Installation

Jika setup script gagal, install manual:

```bash
# Buat virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install dependencies
pip install -r requirements.txt

# Test installation
python3 -c "import tensorflow; import cv2; import streamlit; print('All packages imported successfully')"
```

## 📦 Dependencies

### Core Requirements
- **Python 3.8+**
- **TensorFlow 2.13.0+** - Deep learning framework
- **Streamlit 1.25.0+** - Web interface
- **OpenCV 4.8.0+** - Computer vision
- **NumPy 1.24.0+** - Numerical computing

### Optional Dependencies
- **MTCNN 0.1.1+** - Face detection (fallback jika tidak tersedia)
- **Scikit-learn 1.3.0+** - Machine learning utilities
- **XGBoost 1.7.0+** - Ensemble methods
- **Plotly 5.15.0+** - Interactive visualizations

## 🎮 Cara Penggunaan

### 1. Upload Image
- Klik "Choose an image file"
- Pilih gambar dengan format JPG, PNG, atau BMP
- Pastikan gambar memiliki wajah yang jelas

### 2. Konfigurasi Model
- Pilih model yang ingin digunakan di sidebar
- Atur confidence threshold
- Enable/disable face detection

### 3. Analisis Image
- Klik "🚀 Analyze Image"
- Tunggu proses analisis selesai
- Lihat hasil prediksi dan analisis

### 4. Interpretasi Hasil
- **REAL**: Gambar dianggap asli
- **FAKE**: Gambar dianggap deepfake
- **Confidence Score**: Tingkat kepercayaan (0-1)
- **Consensus**: Persentase model yang setuju

## 🔍 Model Details

### MesoNet (BCN + AGLU)
- **Purpose**: Specialized deepfake detection
- **Enhancement**: Batch Channel Normalization + Adaptive Gaussian Linear Unit
- **Strengths**: Lightweight, fast inference, designed for deepfakes
- **Best for**: Quick screening, mobile deployment

### MobileViT XXS
- **Purpose**: Mobile-optimized Vision Transformer
- **Architecture**: Hybrid CNN-Transformer
- **Strengths**: Balance between accuracy and efficiency
- **Best for**: Resource-constrained environments

### Additional Models
- **EfficientNet B0**: Balanced accuracy-efficiency ratio
- **ResNet50**: Proven deep architecture
- **Vision Transformer**: Pure attention mechanism
- **DenseNet121**: Feature reuse architecture
- **InceptionV3**: Multi-scale feature extraction

## 📊 Performance Insights

### Ensemble Strategy
Sistem menggunakan weighted voting dengan bobot:
- MesoNet: 1.2 (highest weight - specialized for deepfakes)
- MobileViT: 1.1 (good balance)
- EfficientNet: 1.0 (baseline)
- ResNet50: 0.9 (general purpose)
- DenseNet: 0.95 (feature-rich)
- InceptionV3: 0.9 (multi-scale)

### Confidence Interpretation
- **> 0.8**: High confidence
- **0.6 - 0.8**: Moderate confidence
- **< 0.6**: Low confidence (manual review recommended)

## 🛠️ Troubleshooting

### Common Issues

#### 1. TensorFlow Import Error
```bash
# Update TensorFlow
pip install --upgrade tensorflow

# For Apple Silicon Macs
pip install tensorflow-macos tensorflow-metal
```

#### 2. OpenCV Import Error
```bash
# Install OpenCV dependencies (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install libgl1-mesa-glx libgthread-2.0-0

# Reinstall OpenCV
pip uninstall opencv-python opencv-contrib-python
pip install opencv-python
```

#### 3. MTCNN Not Working
```bash
# Reinstall MTCNN
pip uninstall mtcnn
pip install mtcnn

# Alternative face detection akan digunakan otomatis
```

#### 4. Streamlit Port Already in Use
```bash
# Gunakan port yang berbeda
streamlit run simple_app.py --server.port 8502
```

#### 5. Memory Issues
- Gunakan `simple_app.py` untuk environment terbatas
- Kurangi jumlah model yang aktif
- Resize gambar ke ukuran lebih kecil

### Performance Optimization

#### CPU-only Environment
- Set `TF_CPP_MIN_LOG_LEVEL=2` untuk mengurangi warnings
- Gunakan fewer models untuk inference yang lebih cepat
- Enable face detection untuk fokus ke area wajah

#### GPU Environment
- Install `tensorflow-gpu` untuk accelerated training
- Batch processing untuk multiple images
- Larger batch sizes untuk better GPU utilization

## 🔬 Technical Architecture

### Model Loading Strategy
1. **Lazy Loading**: Model dimuat saat pertama kali digunakan
2. **Caching**: Model di-cache untuk penggunaan berikutnya
3. **Fallback**: Demo models jika trained models tidak tersedia
4. **Error Handling**: Graceful degradation jika model loading gagal

### Image Processing Pipeline
1. **Upload & Validation**: Validasi format dan ukuran gambar
2. **Face Detection**: MTCNN atau fallback center crop
3. **Preprocessing**: Resize, normalization, batch preparation
4. **Model Inference**: Parallel prediction dari multiple models
5. **Ensemble Voting**: Weighted combination of predictions
6. **Result Analysis**: Statistical analysis dan visualization

### Security Considerations
- **Input Validation**: Validasi tipe file dan ukuran
- **Memory Management**: Cleanup setelah processing
- **Error Isolation**: Model errors tidak crash aplikasi
- **Resource Limits**: Timeout untuk inference yang lama

## 🚦 System Status Indicators

### Model Status
- ✅ **Available**: Model loaded dan siap digunakan
- 🔧 **Demo Mode**: Menggunakan simulated model
- ❌ **Not Available**: Model gagal dimuat

### Face Detection Status
- ✅ **MTCNN**: Menggunakan MTCNN face detector
- 🔧 **Basic**: Menggunakan center crop fallback
- ❌ **Disabled**: Face detection dinonaktifkan

## 📈 Future Enhancements

### Planned Features
- [ ] **Video Processing**: Support untuk file video
- [ ] **Batch Processing**: Upload multiple images
- [ ] **API Endpoint**: RESTful API untuk integration
- [ ] **Model Training**: Interface untuk fine-tuning
- [ ] **Result Export**: Export hasil ke PDF/CSV

### Model Improvements
- [ ] **Custom Training**: Train model dengan dataset lokal
- [ ] **Transfer Learning**: Fine-tune pre-trained models
- [ ] **Ensemble Optimization**: Automatic weight optimization
- [ ] **Model Compression**: Smaller models untuk mobile

## 📞 Support & Contribution

### Issues
Jika mengalami masalah:
1. Check troubleshooting section di atas
2. Verify system requirements
3. Test dengan `simple_app.py` terlebih dahulu
4. Check console output untuk error messages

### Performance Reporting
Untuk melaporkan performa atau akurasi:
1. Include image yang ditest (jika memungkinkan)
2. Screenshot hasil prediksi
3. System specifications
4. Model configuration yang digunakan

## 📄 License & Citation

Sistem ini dikembangkan untuk penelitian skripsi. Jika menggunakan dalam penelitian:

```
@misc{deepfake_detection_system,
  title={Advanced Multi-Model Deepfake Detection System},
  author={[Your Name]},
  year={2024},
  note={Sistem deteksi deepfake dengan ensemble voting}
}
```

## 🎯 Summary

Sistem Advanced Deepfake Detection ini dirancang khusus untuk mengatasi masalah akurasi rendah pada implementasi MesoNet dan MobileViT tunggal. Dengan menggunakan ensemble dari multiple models dan weighted voting, sistem ini memberikan:

- **Akurasi Lebih Tinggi**: Consensus dari multiple models
- **Robustness**: Fallback mechanisms untuk berbagai kondisi
- **User-Friendly**: Interface yang mudah digunakan
- **Comprehensive Analysis**: Analisis mendalam dengan visualisasi
- **Scalable**: Dapat ditambah model baru dengan mudah

**Happy detecting! 🚀🤖**