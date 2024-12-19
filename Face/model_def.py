import tensorflow as tf
from tensorflow.keras import layers, models
from keras_facenet import FaceNet

INPUT_SHAPE = (220, 220, 3)
EMBEDDING_DIM = 128

facenet_model = FaceNet()


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
