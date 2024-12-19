import numpy as np
from keras_facenet import FaceNet
from preprocess import preprocess_image

facenet = FaceNet()

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


