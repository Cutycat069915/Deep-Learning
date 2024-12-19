import os
import cv2
import numpy as np
from keras_facenet import FaceNet
from mtcnn import MTCNN


class FaceComparer:
    def __init__(self):
        # 初始化 MTCNN 檢測器和 FaceNet 模型
        self.detector = MTCNN()
        self.facenet = FaceNet()

    def preprocess_image(self, image_path):
        """預處理圖片"""
        try:
            # 讀取圖片
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"無法讀取圖片: {image_path}")

            # 轉換為RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # 檢測人臉
            faces = self.detector.detect_faces(img_rgb)
            if not faces:
                raise ValueError(f"未檢測到人臉: {image_path}")

            # 獲取第一個人臉
            face = faces[0]
            x, y, width, height = face['box']

            # 裁剪人臉區域
            face_img = img_rgb[y:y + height, x:x + width]

            # 調整大小
            face_resized = cv2.resize(face_img, (160, 160))

            return face_resized

        except Exception as e:
            print(f"處理圖片時出錯 {image_path}: {str(e)}")
            return None

    def generate_embedding(self, image_path):
        """生成人臉嵌入向量"""
        face = self.preprocess_image(image_path)
        if face is None:
            return None

        # 擴展維度以符合模型輸入
        face_batch = np.expand_dims(face, axis=0)

        # 生成嵌入
        embedding = self.facenet.embeddings(face_batch)
        return embedding[0]

    def compare_faces(self, image1_path, image2_path):
        """比較兩張人臉圖片的相似度"""
        # 提取特徵
        embedding1 = self.generate_embedding(image1_path)
        embedding2 = self.generate_embedding(image2_path)

        if embedding1 is None or embedding2 is None:
            return None

        # 計算餘弦相似度
        similarity = np.dot(embedding1, embedding2)
        return similarity


def main():
    # 創建人臉比較器
    comparer = FaceComparer()

    # 測試圖片路徑 (請替換為你的實際圖片路徑)
    test_image1 = "D:/Desktop/Pic/data/anchor05.jpg"
    test_image2 = "D:/Desktop/Pic/data/negative04.jpg"

    try:
        # 比較人臉
        similarity = comparer.compare_faces(test_image1, test_image2)

        if similarity is not None:
            print(f"人臉相似度: {similarity:.4f}")
            print(f"相似度閾值建議: 0.7 (大於此值可能是同一個人)")
        else:
            print("無法比較人臉")

    except Exception as e:
        print(f"發生錯誤: {str(e)}")


if __name__ == "__main__":
    main()
