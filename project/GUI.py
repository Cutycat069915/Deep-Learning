import os
import numpy as np
import mysql.connector
import keras.models
from Data_preprocess import preprocess_image
from tkinter import Tk, Label, Button, filedialog, messagebox


class FaceRecognitionDatabase:
    def __init__(self, model_path, db_config):
        # 加載模型
        self.model = keras.models.load_model(model_path)

        # 連接MySQL
        self.conn = mysql.connector.connect(**db_config)
        self.cursor = self.conn.cursor()

        # 創建人臉特徵表
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS face_embeddings (
                id INT AUTO_INCREMENT PRIMARY KEY,
                name VARCHAR(255),
                embedding BLOB,
                image_path VARCHAR(255)
            )
        ''')
        self.conn.commit()

    def generate_embedding(self, image_path):
        """生成人臉嵌入向量"""
        face = preprocess_image(image_path)
        face_batch = np.expand_dims(face, axis=0)
        embedding = self.model.predict(face_batch)[0]
        return embedding

    def store_face_embedding(self, name, image_path):
        """將人臉特徵存入資料庫"""
        embedding = self.generate_embedding(image_path)
        embedding_bytes = embedding.tobytes()

        self.cursor.execute('''
            INSERT INTO face_embeddings (name, embedding, image_path) 
            VALUES (%s, %s, %s)
        ''', (name, embedding_bytes, image_path))
        self.conn.commit()

    def find_matching_face(self, input_image_path, threshold=0.6):
        """在資料庫中尋找相似人臉"""
        input_embedding = self.generate_embedding(input_image_path)

        # 查詢所有已存儲的嵌入向量
        self.cursor.execute('SELECT id, name, embedding, image_path FROM face_embeddings')
        stored_faces = self.cursor.fetchall()

        best_match = None
        best_similarity = 0

        for face_id, name, embedding_bytes, image_path in stored_faces:
            # 還原嵌入向量
            stored_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)

            # 計算相似度
            similarity = np.dot(input_embedding, stored_embedding)

            if similarity > best_similarity and similarity > threshold:
                best_match = {
                    'id': face_id,
                    'name': name,
                    'similarity': similarity,
                    'image_path': image_path
                }
                best_similarity = similarity

        return best_match

    def close_connection(self):
        """關閉資料庫連接"""
        self.cursor.close()
        self.conn.close()


class FaceRecognitionApp:
    def __init__(self, root, face_db):
        self.root = root
        self.face_db = face_db
        self.root.title("人臉辨識系統")

        # 介面元件
        self.label = Label(root, text="上傳圖片以進行辨識")
        self.label.pack(pady=20)

        self.upload_button = Button(root, text="上傳圖片", command=self.upload_image)
        self.upload_button.pack(pady=10)

        self.result_label = Label(root, text="", fg="blue")
        self.result_label.pack(pady=20)

    def upload_image(self):
        """用戶上傳圖片並執行人臉辨識"""
        file_path = filedialog.askopenfilename(
            title="選擇圖片",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")]
        )

        if file_path:
            try:
                match = self.face_db.find_matching_face(file_path)

                if match:
                    self.result_label.config(
                        text=f"找到相似人臉: {match['name']}\n相似度: {match['similarity']:.4f}"
                    )
                else:
                    self.result_label.config(text="未找到匹配人臉")
            except Exception as e:
                messagebox.showerror("錯誤", f"處理圖片時發生錯誤：{e}")


def main():
    # 配置
    model_path = r"C:\Users\User\PycharmProjects\Face\Project\Final Project\face_recognition_models\refined_embedding_model.h5"
    db_config = {
        'host': '127.0.0.1',
        'user': 'root',
        'password': '1234',
        'database': 'face_recognition_db'
    }

    # 初始化資料庫
    face_db = FaceRecognitionDatabase(model_path, db_config)

    try:
        # 啟動GUI
        root = Tk()
        app = FaceRecognitionApp(root, face_db)
        root.mainloop()
    finally:
        face_db.close_connection()


if __name__ == "__main__":
    main()





