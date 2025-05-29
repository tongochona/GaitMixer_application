# GaitMixer Application

This repository contains PyTorch code for a gait recognition application, adapted and fine-tuned for a custom dataset.

---

### 🔗 Based on Original Work
Original project: [GaitMixer by exitudio](https://github.com/exitudio/GaitMixer)  

---

## 📹 Demo
<p align="center"><img src="assets/output.gif" width="60%" alt="" /></p>

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
  --train_data_path ../myData/yolo_train.csv \
  --valid_data_path ../myData/yolo_val.csv \
  --debug false \
  --save_model true \
  --weight_path ../../GaitMixer.pt \
  --epochs 30 \
  --name <finetune_model_name> \
  --project <name_project> \
  --learning_rate 6e-4
```
---
### 🧪 Testing
Use the fine-tuned model to recognize individuals from a test video:
```bash
python test.py YOLO \
  --predict_data_path ../video/video_test.MOV \
  --ref_data_path ../myData/train_val.csv \
  --weight_path <path_your_finetune_model>
```
---
## 🛠️ Preprocessing
You can extract pose keypoints from your video using the preprocessing tools available in the *tools/* directory.

