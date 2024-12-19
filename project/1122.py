import os
import random
import cv2
from mtcnn import MTCNN
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# 數據集路徑
data_folder = "D:/Desktop/Pic/data/"
detector = MTCNN()


def align_face(img, keypoints):
    left_eye = keypoints['left_eye']
    right_eye = keypoints['right_eye']

    # 計算兩眼之間的角度
    dY = right_eye[1] - left_eye[1]
    dX = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dY, dX))

    # 中心點為雙眼中點
    eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

    # 生成旋轉矩陣
    M = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)

    # 應用旋轉矩陣
    aligned_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC)
    return aligned_img


def preprocess_and_align(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")

        # 轉換為RGB（因為MTCNN期望RGB格式）
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(img_rgb)

        if not faces:
            print(f"No faces detected in {image_path}")
            return None

        face = faces[0]  # 只處理第一個檢測到的臉
        keypoints = face['keypoints']
        aligned_img = align_face(img, keypoints)

        # 裁剪和縮放
        x, y, width, height = face['box']
        face_img = aligned_img[y:y + height, x:x + width]
        face_resized = cv2.resize(face_img, (160, 160))

        # 歸一化到 [0, 1]
        face_normalized = face_resized.astype(np.float32) / 255.0

        return face_normalized

    except Exception as e:
        print(f"Error processing image {image_path}: {str(e)}")
        return None


# 測試圖片處理
def process_test_image(image_path):
    try:
        # 預處理單張測試圖片
        face = preprocess_and_align(image_path)
        if face is None:
            raise ValueError("Failed to process test image")

        # 添加批次維度
        face_batch = np.expand_dims(face, axis=0)
        return face_batch

    except Exception as e:
        print(f"Error processing test image: {str(e)}")
        return None


# 數據預處理與生成三元組
def generate_triplets(data_folder):
    people = os.listdir(data_folder)
    triplets = []

    for person in people:
        person_path = os.path.join(data_folder, person)

        # 檢查目錄
        anchor_dir = os.path.join(person_path, 'anchor')
        positive_dir = os.path.join(person_path, 'positive')
        negative_dir = os.path.join(person_path, 'negative')

        if not all(os.path.exists(d) for d in [anchor_dir, positive_dir, negative_dir]):
            continue

        # 獲取圖片列表
        anchor_images = [f for f in os.listdir(anchor_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        positive_images = [f for f in os.listdir(positive_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        negative_images = [f for f in os.listdir(negative_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

        if not all([anchor_images, positive_images, negative_images]):
            continue

        # 生成三元組
        for anchor_img in anchor_images:
            for positive_img in positive_images:
                anchor_img_path = os.path.join(anchor_dir, anchor_img)
                positive_img_path = os.path.join(positive_dir, positive_img)
                negative_img_path = os.path.join(negative_dir, random.choice(negative_images))

                # 預處理圖片
                anchor = preprocess_and_align(anchor_img_path)
                positive = preprocess_and_align(positive_img_path)
                negative = preprocess_and_align(negative_img_path)

                # 檢查所有圖片是否成功處理
                if all(img is not None for img in [anchor, positive, negative]):
                    triplets.append((anchor, positive, negative))

    if not triplets:
        raise ValueError("No valid triplets generated")

    return np.array(triplets)


# 嵌入模型
def create_embedding_model(input_shape=(160, 160, 3), embedding_dim=128):
    inputs = layers.Input(shape=input_shape)

    # 卷積層塊
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.GlobalAveragePooling2D()(x)

    # 嵌入層
    embeddings = layers.Dense(embedding_dim, activation=None, name="embeddings")(x)

    # L2 歸一化
    outputs = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))(embeddings)

    model = models.Model(inputs, outputs, name="embedding_model")
    return model


# 三元組損失函數
def triplet_loss(y_true, y_pred, alpha=0.2):
    # 分離 anchor, positive, negative 的嵌入向量
    anchor, positive, negative = y_pred[:, 0:128], y_pred[:, 128:256], y_pred[:, 256:384]

    # 距離計算
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)

    # 損失計算
    loss = tf.maximum(pos_dist - neg_dist + alpha, 0.0)
    return tf.reduce_mean(loss)


# 三元組訓練模型
def create_triplet_model(embedding_model):
    input_anchor = layers.Input(shape=(160, 160, 3), name="anchor")
    input_positive = layers.Input(shape=(160, 160, 3), name="positive")
    input_negative = layers.Input(shape=(160, 160, 3), name="negative")

    # 提取嵌入向量
    anchor_embedding = embedding_model(input_anchor)
    positive_embedding = embedding_model(input_positive)
    negative_embedding = embedding_model(input_negative)

    # 拼接嵌入向量
    concatenated = layers.Concatenate(axis=-1)(
        [anchor_embedding, positive_embedding, negative_embedding]
    )

    # 定義模型
    model = models.Model(
        inputs=[input_anchor, input_positive, input_negative],
        outputs=concatenated,
        name="triplet_model",
    )
    return model


# 主程序
if __name__ == "__main__":
    # 創建和編譯模型
    embedding_model = create_embedding_model()
    triplet_model = create_triplet_model(embedding_model)

    # 編譯三元組模型
    triplet_model.compile(
        optimizer=tf.keras.optimizers.Adam(0.01),
        loss=triplet_loss
    )

    # 測試單張圖片
    test_image_path = "D:/Desktop/Pic/data/anchor05.jpg"
    test_batch = process_test_image(test_image_path)

    if test_batch is not None:
        # 使用嵌入模型生成嵌入向量
        embedding = embedding_model.predict(test_batch)
        print("Embedding shape:", embedding.shape)
        print("Generated Embedding:", embedding[0])
    else:
        print("Failed to process test image")

