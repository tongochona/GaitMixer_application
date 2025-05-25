import sys
import time
import numpy as np
import pandas
import torch
from utils import AverageMeter
from pytorch_metric_learning import distances
import pandas as pd
import torch.nn.functional as F

def _evaluate_casia_b(embeddings):
    """
    Test dataset consists of sequences of last 50 ids from CASIA B Dataset.
    Data is divided in the following way:
    Gallery Set:
        NM 1, NM 2, NM 3, NM 4
    Probe Set:
        Subset 1:
            NM 5, NM 6
         Subset 2:
            BG 1, BG 2
         Subset 3:
            CL 1, CL 2
    """

    gallery = {k: v for (k, v) in embeddings.items()
               if k[1] == 0 and k[2] <= 4}
    gallery_per_angle = {}
    for angle in range(0, 181, 18):
        gallery_per_angle[angle] = {
            k: v for (k, v) in gallery.items() if k[3] == angle}

    probe_nm = {k: v for (k, v) in embeddings.items()
                if k[1] == 0 and k[2] >= 5}
    probe_bg = {k: v for (k, v) in embeddings.items() if k[1] == 1}
    probe_cl = {k: v for (k, v) in embeddings.items() if k[1] == 2}

    correct = np.zeros((3, 11, 11))
    total = np.zeros((3, 11, 11))
    for gallery_angle in range(0, 181, 18):
        gallery_embeddings = np.array(
            list(gallery_per_angle[gallery_angle].values()))
        gallery_targets = list(gallery_per_angle[gallery_angle].keys())
        gallery_pos = int(gallery_angle / 18)

        probe_num = 0
        for probe in [probe_nm, probe_bg, probe_cl]:
            for (target, embedding) in probe.items():
                subject_id, _, _, probe_angle = target
                probe_pos = int(probe_angle / 18)

                # l2 distance
                # distance = np.linalg.norm(gallery_embeddings - embedding, ord=2, axis=1)
                # cosine distance
                gallery_embeddings_norm = gallery_embeddings / \
                    np.linalg.norm(gallery_embeddings, ord=2,
                                   axis=1, keepdims=True)
                embedding_norm = embedding / \
                    np.linalg.norm(embedding, ord=2, keepdims=True)
                distance = 1 - gallery_embeddings_norm @ embedding_norm

                min_pos = np.argmin(distance)
                min_target = gallery_targets[int(min_pos)]

                if min_target[0] == subject_id:
                    correct[probe_num, gallery_pos, probe_pos] += 1
                total[probe_num, gallery_pos, probe_pos] += 1

            probe_num += 1

    accuracy = correct / total

    # Exclude same view
    for i in range(3):
        accuracy[i] -= np.diag(np.diag(accuracy[i]))

    accuracy_flat = np.sum(accuracy, 1) / 10

    header = ["NM#5-6", "BG#1-2", "CL#1-2"]

    sub_accuracies_avg = np.mean(accuracy_flat, 1)
    # accuracy_avg = np.mean(accuracy)
    accuracy_avg = np.mean(sub_accuracies_avg)
    sub_accuracies = dict(zip(header, list(sub_accuracies_avg)))

    dataframe = pandas.DataFrame(
        np.concatenate(
            (accuracy_flat, sub_accuracies_avg[..., np.newaxis]), 1),
        header,
        list(range(0, 181, 18)) + ["mean"],
    )
    print(dataframe)
    return accuracy_avg, sub_accuracies

def rm_same_view(tb):
        tb = tb[:-1, :-1]
        return (tb - torch.diag(torch.diag(tb)))

def _evaluate_OUMVLP(embeddings):
    _distance = distances.CosineSimilarity()

    angles = list(range(0, 91, 15)) + list(range(180, 271, 15))
    num_angles = len(angles)
    gallery = {k: v for (k, v) in embeddings.items() if k[2] == 1}

    gallery_per_angle = {}
    for angle in angles:
        gallery_per_angle[angle] = {k: v for (k, v) in gallery.items() if k[1] == angle}

    probe = {k: v for (k, v) in embeddings.items() if k[2] == 0}

    accuracy = torch.zeros((num_angles + 1, num_angles + 1))
    correct = torch.zeros_like(accuracy)
    total = torch.zeros_like(accuracy)

    for gallery_angle in angles:
        gallery_embeddings = torch.stack(list(gallery_per_angle[gallery_angle].values()), 0)
        gallery_targets = list(gallery_per_angle[gallery_angle].keys())
        gallery_pos = angles.index(gallery_angle)

        probe_embeddings = torch.stack(list(probe.values()))
        q_g_dist = 1-_distance(probe_embeddings, gallery_embeddings)
        for idx, target in enumerate(probe.keys()):
            subject_id, probe_angle, _ = target
            
            # rm non-exist
            if True: #(subject_id, gallery_angle, 1) in gallery: #
                probe_pos = angles.index(probe_angle)

                min_pos = torch.argmin(q_g_dist[idx])
                min_target = gallery_targets[int(min_pos)]

                if min_target[0] == subject_id:
                    correct[gallery_pos, probe_pos] += 1
                total[gallery_pos, probe_pos] += 1

    accuracy[:-1, :-1] = correct[:-1, :-1] / total[:-1, :-1]

    accuracy[:-1, -1] = torch.mean(accuracy[:-1, :-1], dim=1)
    accuracy[-1, :-1] = torch.mean(accuracy[:-1, :-1], dim=0)

    accuracy_avg = torch.mean(accuracy[:-1, :-1])
    accuracy[-1, -1] = accuracy_avg
    df = pd.DataFrame(
        accuracy.numpy(),
        angles + ["mean"],
        angles + ["mean"],
    )
    df = (df * 100).round(1)

    print(f"accuracy: {accuracy_avg * 100:.1f} %")
    # print(df.to_latex())

    # acc_no_iden = rm_same_view(correct).sum()/rm_same_view(total).sum()
    acc_no_iden = rm_same_view(accuracy[:-1, :-1]).mean()
    print('rm same view', acc_no_iden)
    print(df)
    ############################

    return acc_no_iden, {'avg': acc_no_iden}


def _evaluate_YOLO(embeddings):
    """
    Đánh giá độ chính xác của YOLOPose dựa trên khoảng cách Cosine.
    Chia gallery và probe theo video (gallery: video đầu, probe: các video còn lại).
    
    embeddings: dict {(person_id, video_name, first_frame, last_frame): embedding_vector}
    """
    _distance = F.cosine_similarity  # Dùng hàm torch

    # Nhóm embeddings theo person_id
    split_ratio = 0.8  

    embeddings_by_person = {}
    for key, emb in embeddings.items():
        person_id, video_name, first_frame, last_frame = key
        if person_id not in embeddings_by_person:
            embeddings_by_person[person_id] = []
        embeddings_by_person[person_id].append((key, emb))

    gallery = {}
    probe = {}

    for person_id, items in embeddings_by_person.items():
        if len(items) < 2:
            print(f"Warning: Person {person_id} has only one embedding. Skipping...")
            continue

        # Chia theo tỉ lệ
        split_index = int(len(items) * split_ratio)
        gallery_items = items[:split_index]
        probe_items = items[split_index:]

        # Đảm bảo mỗi người có ít nhất một mẫu trong cả gallery và probe
        if len(gallery_items) == 0 or len(probe_items) == 0:
            print(f"Warning: Person {person_id} doesn't have enough samples after split. Skipping...")
            continue

        for key, emb in gallery_items:
            gallery[key] = emb
        for key, emb in probe_items:
            probe[key] = emb

    if not gallery or not probe:
        raise ValueError("Không đủ dữ liệu để chia gallery và probe.")

    # Chuyển sang tensor
    gallery_ids = list(gallery.keys())
    probe_ids = list(probe.keys())
    gallery_embeddings = torch.stack([torch.tensor(gallery[i]) for i in gallery_ids])
    probe_embeddings = torch.stack([torch.tensor(probe[i]) for i in probe_ids])

    correct = 0
    total = 0

    # print("\n===== DỰ ĐOÁN KẾT QUẢ =====")
    # print("Probe ID (Thực)    |  Predicted ID (Dự đoán)")

    for probe_idx, probe_id in enumerate(probe_ids):
        similarity_scores = _distance(probe_embeddings[probe_idx].unsqueeze(0), gallery_embeddings, dim=1)
        min_pos = torch.argmax(similarity_scores)  # Max vì cosine similarity càng cao càng tốt
        predicted_id = gallery_ids[min_pos]
        
        # print(f"{str(probe_id[0]):<20} |  {str(predicted_id[0]):<20}")

        if predicted_id[0] == probe_id[0]:
            correct += 1
        total += 1

    accuracy_avg = correct / total if total > 0 else 0
    # print(f"\n🔹 Accuracy: {accuracy_avg * 100:.2f}% 🔹")

    return accuracy_avg, accuracy_avg

def evaluate(data_loader, model, evaluation_fn, gpu=False):
    model.eval()
    batch_time = AverageMeter()
    use_flip=True

    # Calculate embeddings
    with torch.no_grad():
        end = time.time()
        embeddings = dict()
        for idx, (points, target) in enumerate(data_loader):
            is_3seq = False
            if isinstance(points, list):
                is_3seq = True
            if is_3seq and not use_flip:
                raise ValueError(
                    "Average 3 Seq without using flip is not supported")
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
                    f6 = torch.split(
                        output, [bsz, bsz, bsz, bsz, bsz, bsz], dim=0)
                    output = torch.mean(torch.stack(f6), dim=0)
                else:
                    f1, f2 = torch.split(output, [bsz, bsz], dim=0)
                    output = torch.mean(torch.stack([f1, f2]), dim=0)

            for i in range(output.shape[0]):
                sequence = tuple(
                    int(t[i]) if type(t[i]) is torch.Tensor else t[i] for t in target
                )
                if gpu:
                    embeddings[sequence] = output[i]
                else:
                    embeddings[sequence] = output[i].cpu().numpy()

            batch_time.update(time.time() - end)
            end = time.time()

            if idx % 10 == 0:
                print(
                    f"Test: [{idx}/{len(data_loader)}]\t"
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                )
                sys.stdout.flush()

    return evaluation_fn(embeddings)

