import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from preprocess import preprocess_image, generate_triplets
from model_def import create_triplet_model, create_refined_embedding_model, triplet_loss

INPUT_SHAPE = (220, 220, 3)
EMBEDDING_DIM = 128
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001

data_folder = "D:/Desktop/Pic/data/"

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






