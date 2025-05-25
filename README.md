# GaitMixer Application

This repository contains PyTorch code for a gait recognition application, adapted and fine-tuned for a custom dataset.

---

### 🔗 Based on Original Work
Original project: [GaitMixer by exitudio](https://github.com/exitudio/GaitMixer)  

---

## 📹 Demo
[Video demo](assets/output.mp4)

---

## 🚀 Quick Start

### 📁 Data Preparation
You can download the sample dataset and test video from [this link](https://drive.google.com/drive/folders/1ksTNaDQcywfT-sLc5N7luXUwR3l-L_Gq?usp=sharing).

---

### 🧠 Model Preparation
- Download the pre-trained model and set up the environment as instructed in the [original GaitMixer repository](https://github.com/exitudio/GaitMixer).
- Additionally, we use [YOLO Pose Estimation](https://docs.ultralytics.com/tasks/pose/) for keypoint extraction.

---

### 🔧 Finetuning

Fine-tune the original GaitMixer model using your own dataset:

```bash
python train.py YOLO \
  --train_data_path ../myData/keypoints_conf.csv \
  --valid_data_path ../myData/keypoints.csv \
  --debug false \
  --save_model true \
  --weight_path ../../GaitMixer.pt \
  --epochs 50 \
  --name <finetune_model_name>
```
---
### 🧪 Testing
Use the fine-tuned model to recognize individuals from a test video:
```bash
python test.py YOLO \
  --predict_data_path ../video/video_test.MOV \
  --ref_data_path ../myData/keypoints.csv \
  --weight_path <path_your_finetune_model>
```
---
## 🛠️ Preprocessing
You can extract pose keypoints from your video using the preprocessing tools available in the *tools/* directory.

