import os
from ultralytics import YOLO
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from datasets.augmentation import *
from common import *
from models.SpatialTransformerTemporalConv import SpatialTransformerTemporalConv, SpatioTemporalTransformer
from datasets.gait import YOLOPose
from collections import defaultdict
from scipy.spatial.distance import euclidean
import time

def predict_identity(model, probe, gallery, threshold=0.7):
    """
    Nhận diện danh tính bằng cách so sánh embedding mới với gallery.
    Trả về -1 nếu độ tương đồng cao nhất nhỏ hơn ngưỡng.
    """
    _distance = F.cosine_similarity 
    embeddings = gallery
    gallery_ids = list(embeddings.keys())
    gallery_embeddings = torch.stack([torch.tensor(embeddings[i]) for i in gallery_ids])
    predict_data = probe
    input_data = torch.tensor(predict_data, dtype=torch.float32).unsqueeze(0)
    if torch.cuda.is_available():
        input_data = input_data.cuda()
    model.eval()
    with torch.no_grad():
        new_embedding = model(input_data).cpu()
        similarity_scores = _distance(new_embedding, gallery_embeddings, dim=1)
        max_score, min_pos = torch.max(similarity_scores, dim=0)
        if max_score.item() < threshold:
            print(f"Max similarity score: {max_score.item():.4f}, Predicted: -1 (Unknown)")
            return -1
        predicted_id = gallery_ids[min_pos]
        print(f"Max similarity score: {max_score.item():.4f}, Predicted ID: {predicted_id[0]}")
        return predicted_id[0]

def get_embedding(data_loader, model, gpu=False):
    model.eval()
    use_flip=True
    with torch.no_grad():
        embeddings = dict()
        for idx, (points, target) in enumerate(data_loader):
            is_3seq = False
            if isinstance(points, list):
                is_3seq = True
            if is_3seq and not use_flip:
                raise ValueError("Average 3 Seq without using flip is not supported")
            if use_flip:
                if isinstance(points, list):
                    bsz = points[0].shape[0]
                    data_flipped0 = torch.flip(points[0], dims=[1])
                    data_flipped1 = torch.flip(points[1], dims=[1])
                    data_flipped2 = torch.flip(points[2], dims=[1])
                    points = torch.cat(
                        [points[0], data_flipped0, points[1], data_flipped1, points[2], data_flipped2], dim=0)
                else:
                    bsz = points.shape[0]
                    data_flipped = torch.flip(points, dims=[1])
                    points = torch.cat([points, data_flipped], dim=0)
            if torch.cuda.is_available():
                points = points.cuda(non_blocking=True)
            output = model(points)
            if use_flip:
                if is_3seq:
                    f6 = torch.split(output, [bsz, bsz, bsz, bsz, bsz, bsz], dim=0)
                    output = torch.mean(torch.stack(f6), dim=0)
                else:
                    f1, f2 = torch.split(output, [bsz, bsz], dim=0)
                    output = torch.mean(torch.stack([f1, f2]), dim=0)
            for i in range(output.shape[0]):
                sequence = tuple(int(t[i]) if type(t[i]) is torch.Tensor else t[i] for t in target)
                if gpu:
                    embeddings[sequence] = output[i]
                else:
                    embeddings[sequence] = output[i].cpu().numpy()
    return embeddings

def main(opt):
    opt = setup_environment(opt)

    ref_transform = transforms.Compose(
        [
            SelectSequenceCenter(opt.sequence_length),
            remove_conf(enable=opt.rm_conf),
            # normalize_width,
            ToTensor()
        ]
    )
    dataset_ref = YOLOPose(
        opt.ref_data_path,
        sequence_length=opt.sequence_length,
        transform=ref_transform
    )
    ref_loader = torch.utils.data.DataLoader(
        dataset_ref,
        batch_size=512,
        num_workers=opt.num_workers,
        pin_memory=True,
    )
    if opt.model_type == "spatialtransformer_temporalconv":
        model = SpatialTransformerTemporalConv(
            num_frame=opt.sequence_length, in_chans=2 if opt.rm_conf else 3, 
            spatial_embed_dim=opt.embedding_spatial_size, out_dim=opt.embedding_layer_size, 
            num_joints=17, kernel_frame=opt.kernel_frame)
    elif opt.model_type == "spatiotemporal_transformer":
        model = SpatioTemporalTransformer(
            num_frame=opt.sequence_length, in_chans=2 if opt.rm_conf else 3, 
            spatial_embed_dim=opt.embedding_spatial_size, out_dim=opt.embedding_layer_size, 
            num_joints=17)
    elif opt.model_type == "gaitgraph":
        model = get_model_resgcn()
    else:
        raise ValueError("No model type support:", opt.model_type)
    
    print("# parameters: ", count_parameters(model))
    load_checkpoint(model, opt)
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    if opt.cuda:
        model.cuda()

    gallery = get_embedding(ref_loader, model, gpu=False)
    input_video_path = opt.predict_data_path
    video_name = os.path.basename(input_video_path)
    output_video_path = f"/content/drive/MyDrive/PBL4/GaitMixer/{os.path.splitext(video_name)[0]}_out.mp4"
    model_yolo = YOLO("/content/drive/MyDrive/PBL4/yolo11n-pose.pt")
    model_yolo.fuse()
    model_yolo.to("cuda" if torch.cuda.is_available() else "cpu") 
    name = {-1: "Unknown", 1: "Dat", 2: "Hoan", 3: "Dinh", 4: "Truong"}
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError("Không thể mở video gốc")
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    # Bắt đầu đo thời gian
    start_time = time.time()
    
    # Theo dõi chuỗi keypoints và ID dự đoán cho mỗi người
    sequences = defaultdict(list)  # {track_id: [(frame_idx, keypoints, bbox), ...]}
    track_ids = {}  # {track_id: id} lưu ID dự đoán gần nhất
    track_counter = 0  # Đếm số track_id
    
    # Định nghĩa skeleton cho YOLO Pose (17 keypoints)
    skeleton = [
        (0, 1), (0, 2),  # Mũi - Mắt trái, Mũi - Mắt phải
        (1, 3), (2, 4),  # Mắt - Tai
        (0, 5), (0, 6),  # Mũi - Vai
        (5, 7), (6, 8),  # Vai - Khuỷu tay
        (7, 9), (8, 10),  # Khuỷu tay - Cổ tay
        (5, 11), (6, 12),  # Vai - Hông
        (11, 12),  # Hông trái - Hông phải
        (11, 13), (12, 14),  # Hông - Đùi
        (13, 15), (14, 16)  # Đùi - Chân
    ]
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame_idx >= total_frames:
            break
        results = model_yolo(frame, verbose=False)
        result = results[0]
        current_detections = []
        if result.keypoints is not None and result.boxes is not None:
            kpts = result.keypoints.xyn.cpu().numpy()  # (N, 17, 2)
            boxes = result.boxes.xyxy.cpu().numpy()    # (N, 4)
            num_people = min(len(kpts), len(boxes))
            for i in range(num_people):
                keypoints = kpts[i].astype(np.float32)  # (17, 2)
                bbox = boxes[i].astype(np.float32)      # (4,)
                current_detections.append((frame_idx, keypoints, bbox))
                # Vẽ keypoints lên khung hình
                for kp in keypoints:
                    x, y = kp
                    if x > 0 and y > 0:  # Chỉ vẽ nếu tọa độ hợp lệ
                        pixel_x = int(x * width)
                        pixel_y = int(y * height)
                        cv2.circle(frame, (pixel_x, pixel_y), 3, (0, 0, 255), -1)  # Đỏ, bán kính 3
                # Vẽ skeleton (nối keypoints) màu xanh dương
                for (i, j) in skeleton:
                    x1, y1 = keypoints[i]
                    x2, y2 = keypoints[j]
                    if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
                        cv2.line(frame, (int(x1 * width), int(y1 * height)),
                                 (int(x2 * width), int(y2 * height)), (255, 0, 0), 1)  # Xanh dương
        # Gán track_id cho các phát hiện
        if frame_idx == 0:
            for det in current_detections:
                sequences[track_counter].append(det)
                track_counter += 1
        else:
            unmatched_detections = current_detections[:]
            for track_id in list(sequences.keys()):
                if not sequences[track_id] or sequences[track_id][-1][0] != frame_idx - 1:
                    del sequences[track_id]
                    track_ids.pop(track_id, None)
                    continue
                last_bbox = sequences[track_id][-1][2]
                last_center = ((last_bbox[0] + last_bbox[2]) / 2, (last_bbox[1] + last_bbox[3]) / 2)
                min_dist = float('inf')
                best_det = None
                best_det_idx = -1
                for idx, det in enumerate(unmatched_detections):
                    bbox = det[2]
                    center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
                    dist = euclidean(last_center, center)
                    if dist < min_dist:
                        min_dist = dist
                        best_det = det
                        best_det_idx = idx
                if best_det and min_dist < 100:
                    sequences[track_id].append(best_det)
                    unmatched_detections.pop(best_det_idx)
                else:
                    del sequences[track_id]
                    track_ids.pop(track_id, None)
            for det in unmatched_detections:
                sequences[track_counter].append(det)
                track_counter += 1
        # Kiểm tra và dự đoán cho các chuỗi đủ 60 khung hình, chỉ tại các khung cách nhau 5
        if frame_idx >= 60 and frame_idx % 10 == 0:
            for track_id in list(sequences.keys()):
                seq = sequences[track_id]
                if len(seq) >= 60:
                    kpts_seq = np.stack([det[1] for det in seq[-60:]], axis=0)/1080  # (60, 17, 2) normalize
                    if kpts_seq.shape == (60, 17, 2):
                        id = predict_identity(model, kpts_seq, gallery)
                        track_ids[track_id] = id  # Lưu ID dự đoán
                        print(f"Detection at frame {frame_idx} (track {track_id}):")
                        print(f"  🆔 Dự đoán: {id} ({name.get(id, f'Unknown ({id})')})")
        # Vẽ bounding box và tên cho tất cả phát hiện nếu có ID
        for track_id in sequences:
            seq = sequences[track_id]
            if seq and track_id in track_ids:
                latest_det = seq[-1]
                bbox = latest_det[2]
                x_min, y_min, x_max, y_max = map(int, bbox)
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                x_max = min(width, x_max)
                y_max = min(height, y_max)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                id = track_ids[track_id]
                name_text = name.get(id, f"Unknown ({id})")
                cv2.putText(frame, f"ID: {name_text}", (x_min, y_min - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        out.write(frame)
        frame_idx += 1
    cap.release()
    out.release()
    
    # Tính và in thời gian xử lý
    end_time = time.time()
    processing_time = end_time - start_time
    print(f"Video đã được lưu tại: {output_video_path}")
    print(f"Thời gian xử lý video: {processing_time:.2f} giây")

if __name__ == "__main__":
    opt = parse_option()
    main(opt)