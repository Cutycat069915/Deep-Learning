import os
import random
import cv2
import numpy as np
from mtcnn import MTCNN

detector = MTCNN()


def preprocess_image(image_path):
    """預處理單張圖片"""
    try:
        # 讀取圖片
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"無法讀取圖片: {image_path}")

        # 轉換為RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 檢測人臉
        faces = detector.detect_faces(img_rgb)
        if not faces:
            raise ValueError(f"未檢測到人臉: {image_path}")

        # 獲取第一個人臉
        face = faces[0]
        x, y, width, height = face['box']

        # 裁剪人臉區域
        face_img = img_rgb[y:y + height, x:x + width]

        # 調整大小
        face_resized = cv2.resize(face_img, (220, 220))

        # 歸一化
        face_normalized = face_resized.astype(np.float32) / 255.0

        return face_normalized

    except Exception as e:
        print(f"處理圖片時出錯 {image_path}: {str(e)}")
        return None


def generate_triplets(data_folder):
    """生成訓練用的三元組"""
    triplets = []
    people = os.listdir(data_folder)

    for person in people:
        person_path = os.path.join(data_folder, person)

        # 檢查必要的子目錄
        anchor_dir = os.path.join(person_path, 'anchor')
        positive_dir = os.path.join(person_path, 'positive')
        negative_dir = os.path.join(person_path, 'negative')

        if not all(os.path.exists(d) for d in [anchor_dir, positive_dir, negative_dir]):
            continue

        # 獲取圖片列表
        anchor_images = [f for f in os.listdir(anchor_dir)
                         if f.endswith(('.jpg', '.jpeg', '.png'))]
        positive_images = [f for f in os.listdir(positive_dir)
                           if f.endswith(('.jpg', '.jpeg', '.png'))]
        negative_images = [f for f in os.listdir(negative_dir)
                           if f.endswith(('.jpg', '.jpeg', '.png'))]

        if not all([anchor_images, positive_images, negative_images]):
            continue

        # 生成三元組
        for anchor_img in anchor_images:
            for positive_img in positive_images:
                anchor_img_path = os.path.join(anchor_dir, anchor_img)
                positive_img_path = os.path.join(positive_dir, positive_img)
                negative_img_path = os.path.join(negative_dir,
                                                 random.choice(negative_images))

                # 預處理圖片
                anchor = preprocess_image(anchor_img_path)
                positive = preprocess_image(positive_img_path)
                negative = preprocess_image(negative_img_path)

                if all(img is not None for img in [anchor, positive, negative]):
                    triplets.append((anchor, positive, negative))

    if not triplets:
        raise ValueError("沒有生成有效的三元組")

    return np.array(triplets)
