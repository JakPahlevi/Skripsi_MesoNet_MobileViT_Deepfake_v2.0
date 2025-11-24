# Enhanced Deepfake Detection - New Features Documentation

## 🎬 Video Preview Feature

### What's New:
- **Video Preview**: Langsung menampilkan video yang di-upload dalam aplikasi
- **Video Information**: Menampilkan detail file (nama, ukuran, tipe)
- **Ground Truth Selection**: Pilihan label actual (Real/Fake) untuk evaluasi

### How to Use:
1. Upload video file
2. Video akan langsung ter-preview di aplikasi
3. Pilih ground truth label (Real atau Fake)
4. Lanjutkan dengan analisis

## 📊 Evaluation Metrics Table

### Metrics Included:
- **Accuracy**: Ketepatan prediksi secara keseluruhan
- **Precision**: Ketepatan prediksi positive (Fake)
- **Recall**: Kemampuan mendeteksi semua Fake videos
- **F1-Score**: Harmonic mean dari Precision dan Recall
- **Loss**: Binary cross-entropy loss
- **Average Confidence**: Rata-rata confidence score

### Models Evaluated:
1. **MesoNet**: Model CNN untuk deteksi deepfake
2. **MobileViT**: Model Vision Transformer mobile
3. **Ensemble**: Kombinasi kedua model

### Detailed Performance Analysis:
- **True Positives**: Fake videos yang benar terdeteksi Fake
- **True Negatives**: Real videos yang benar terdeteksi Real
- **False Positives**: Real videos yang salah terdeteksi Fake
- **False Negatives**: Fake videos yang salah terdeteksi Real

## 🔄 How to Run the Application

### Method 1: Enhanced Launcher (Recommended)
```bash
cd /home/jak/myenv/skripsi_fix/streamlit
./launch_enhanced_fixed.sh
```

### Method 2: Manual Streamlit
```bash
cd /home/jak/myenv/skripsi_fix/streamlit
source venv/bin/activate
streamlit run enhanced_video_app.py --server.port=8506
```

### Method 3: Direct Command
```bash
streamlit run enhanced_video_app.py
```

## ✨ Complete Feature List

### 🎥 Video Processing
- ✅ Video preview in-app
- ✅ Complete frame extraction (no 30-frame limit)
- ✅ Real-time progress tracking
- ✅ Multiple video format support (MP4, AVI, MOV, MKV)

### 🧠 AI Models
- ✅ Enhanced MesoNet with Spatio-Temporal Attention
- ✅ Enhanced MobileViT with Vision Transformer
- ✅ Separate analysis for each model
- ✅ Ensemble prediction combining both models

### 📊 Analysis Results
- ✅ Detailed frame-by-frame results table
- ✅ Evaluation metrics table (Accuracy, Precision, Recall, F1, Loss)
- ✅ Temporal consistency analysis
- ✅ Model comparison analysis
- ✅ Advanced visualizations

### 📈 Visualizations
- ✅ Prediction scores over time
- ✅ Score distribution histograms
- ✅ Model agreement scatter plots
- ✅ Confidence comparison charts

### 💾 Export Features
- ✅ Download detailed results as CSV
- ✅ Download evaluation metrics as CSV
- ✅ Export temporal consistency data

## 🎯 Usage Workflow

1. **Upload Video**: Choose video file and see preview
2. **Set Ground Truth**: Select actual label (Real/Fake)
3. **Processing**: Wait for complete frame analysis
4. **View Metrics**: Check evaluation metrics table
5. **Analyze Results**: Review model comparisons and temporal analysis
6. **Export Data**: Download results and metrics for further analysis

## 🔧 Technical Specifications

### Dependencies Status:
- ✅ TensorFlow 2.16.1
- ✅ MTCNN face detection
- ✅ OpenCV 4.12.0
- ✅ Streamlit 1.49.1

### Performance Features:
- ✅ Spatio-Temporal Consistency algorithms
- ✅ Attention mechanisms implementation
- ✅ Real-time face detection with MTCNN
- ✅ Multi-model ensemble predictions

### Evaluation Capabilities:
- ✅ Confusion matrix calculations
- ✅ Binary classification metrics
- ✅ Model performance comparison
- ✅ Temporal stability assessment

## 🌐 Access URLs

- **Application**: http://localhost:8506
- **Network Access**: http://172.26.202.106:8506

## 📋 System Requirements

- Python 3.12+
- Virtual environment activated
- Minimum 4GB RAM for video processing
- 500MB max upload size for videos

This enhanced version provides comprehensive deepfake detection analysis with professional-grade evaluation metrics and user-friendly video preview capabilities.