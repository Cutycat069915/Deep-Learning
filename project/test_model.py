import os
import keras.models
import numpy as np
from Data_preprocess import preprocess_image

def compare_custom_model_faces(image1_path, image2_path, model):
    # 提取特徵
    face1 = preprocess_image(image1_path)
    face2 = preprocess_image(image2_path)

    # 擴展維度
    face1_batch = np.expand_dims(face1, axis=0)
    face2_batch = np.expand_dims(face2, axis=0)

    # 生成嵌入向量
    features1 = model.predict(face1_batch)
    features2 = model.predict(face2_batch)

    # 計算餘弦相似度
    similarity = np.dot(features1[0], features2[0])
    return similarity


if __name__ == "__main__":
    # 設置模型路徑
    model_path = r"C:\Users\User\PycharmProjects\Face\Project\Final Project\face_recognition_models\refined_embedding_model.h5"
    model = keras.models.load_model(model_path)

    try:
        # 測試圖片
        test_image1 = "D:/Desktop/Pic/data/anchor05.jpg"
        test_image2 = "D:/Desktop/Pic/data/anchor07.jpg"

        similarity = compare_custom_model_faces(test_image1, test_image2, model)
        print(f"人臉相似度: {similarity:.4f}")
        print(f"相似度閾值建議: 0.7 (大於此值可能是同一個人)")

    except Exception as e:
        print(f"發生錯誤: {str(e)}")
