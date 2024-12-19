import os
import random
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from mtcnn import MTCNN
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras_facenet import FaceNet

# 全局配置
INPUT_SHAPE = (160, 160, 3)
EMBEDDING_DIM = 128
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001

# 初始化MTCNN檢測器和FaceNet
detector = MTCNN()
facenet_model = FaceNet()


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
        face_resized = cv2.resize(face_img, (160, 160))

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


def create_refined_embedding_model():
    """基於預訓練FaceNet創建精細化嵌入模型"""
    # 加載預訓練FaceNet模型
    base_model = facenet_model.model

    # 凍結大部分原始層
    for layer in base_model.layers[:-3]:
        layer.trainable = False

    # 添加微調層
    x = base_model.layers[-3].output
    x = layers.Dense(512, activation='relu', name='fine_tune_dense1')(x)
    x = layers.BatchNormalization(name='fine_tune_bn1')(x)
    x = layers.Dropout(0.5, name='fine_tune_dropout')(x)
    x = layers.Dense(EMBEDDING_DIM, name='fine_tune_embedding')(x)
    x = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1),
                      name='l2_normalize')(x)

    refined_model = models.Model(inputs=base_model.input, outputs=x)

    return refined_model


def create_triplet_model(embedding_model):
    """創建三元組模型"""
    # 三個輸入
    input_anchor = layers.Input(shape=INPUT_SHAPE, name='anchor')
    input_positive = layers.Input(shape=INPUT_SHAPE, name='positive')
    input_negative = layers.Input(shape=INPUT_SHAPE, name='negative')

    # 共享權重的特徵提取
    anchor_embedding = embedding_model(input_anchor)
    positive_embedding = embedding_model(input_positive)
    negative_embedding = embedding_model(input_negative)

    # 合併輸出
    merged = layers.Concatenate(axis=-1)([
        anchor_embedding,
        positive_embedding,
        negative_embedding
    ])

    model = models.Model(
        inputs=[input_anchor, input_positive, input_negative],
        outputs=merged,
        name='triplet_model'
    )

    return model


def triplet_loss(y_true, y_pred, alpha=0.2):
    """三元組損失函數"""
    total_length = y_pred.shape.as_list()[-1]

    anchor = y_pred[:, 0:int(total_length * 1 / 3)]
    positive = y_pred[:, int(total_length * 1 / 3):int(total_length * 2 / 3)]
    negative = y_pred[:, int(total_length * 2 / 3):int(total_length)]

    # 計算距離
    pos_dist = tf.reduce_sum(tf.square(anchor - positive), axis=1)
    neg_dist = tf.reduce_sum(tf.square(anchor - negative), axis=1)

    # 基本三元組損失
    basic_loss = tf.maximum(pos_dist - neg_dist + alpha, 0.0)

    return tf.reduce_mean(basic_loss)


def train_model(data_folder, save_dir='models'):
    """訓練模型的主函數"""
    print("開始生成訓練數據...")
    triplets = generate_triplets(data_folder)

    # 分離數據
    anchors = triplets[:, 0]
    positives = triplets[:, 1]
    negatives = triplets[:, 2]

    # 創建驗證集
    indices = np.arange(len(triplets))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    train_anchors, train_positives, train_negatives = (
        anchors[train_idx], positives[train_idx], negatives[train_idx]
    )
    val_anchors, val_positives, val_negatives = (
        anchors[val_idx], positives[val_idx], negatives[val_idx]
    )

    # 創建模型
    embedding_model = create_refined_embedding_model()
    triplet_model = create_triplet_model(embedding_model)

    # 編譯模型
    triplet_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=triplet_loss
    )

    # 準備回調函數
    os.makedirs(save_dir, exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(save_dir, 'best_refined_model.h5'),
            save_best_only=True,
            monitor='val_loss'
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=os.path.join(save_dir, 'logs'),
            histogram_freq=1
        )
    ]

    # 創建虛擬標籤
    dummy_train = np.zeros((len(train_idx), EMBEDDING_DIM * 3))
    dummy_val = np.zeros((len(val_idx), EMBEDDING_DIM * 3))

    # 訓練模型
    print("開始訓練模型...")
    history = triplet_model.fit(
        [train_anchors, train_positives, train_negatives],
        dummy_train,
        validation_data=([val_anchors, val_positives, val_negatives], dummy_val),
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )

    # 保存模型
    embedding_model.save(os.path.join(save_dir, 'refined_embedding_model.h5'))

    # 繪製訓練歷史
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'))
    plt.close()

    return embedding_model, history


def extract_features(image_path, model):
    """使用訓練好的模型提取特徵"""
    face = preprocess_image(image_path)
    if face is None:
        return None

    # 添加批次維度
    face_batch = np.expand_dims(face, axis=0)

    # 提取特徵
    features = model.predict(face_batch)
    return features[0]


def compare_faces(image1_path, image2_path, model):
    """比較兩張人臉圖片的相似度"""
    # 提取特徵
    features1 = extract_features(image1_path, model)
    features2 = extract_features(image2_path, model)

    if features1 is None or features2 is None:
        return None

    # 計算餘弦相似度
    similarity = np.dot(features1, features2)
    return similarity


if __name__ == "__main__":
    # 設置數據路徑
    data_folder = "D:/Desktop/Pic/data/"
    save_dir = "face_recognition_models"

    try:
        # 訓練模型
        embedding_model, history = train_model(data_folder, save_dir)
        print("模型訓練完成！")

        # 測試模型
        test_image1 = "D:/Desktop/Pic/data/anchor05.jpg"
        test_image2 = "D:/Desktop/Pic/data/anchor07.jpg"

        similarity = compare_faces(test_image1, test_image2, embedding_model)
        if similarity is not None:
            print(f"人臉相似度: {similarity:.4f}")
            print(f"相似度閾值建議: 0.7 (大於此值可能是同一個人)")

    except Exception as e:
        print(f"發生錯誤: {str(e)}")

