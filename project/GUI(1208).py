import os
import numpy as np
import mysql.connector
import keras.models
from Data_preprocess import preprocess_image
from tkinter.ttk import Progressbar, Style
from tkinter import Tk, Label, Button, filedialog, messagebox
from customtkinter import *
from PIL import Image, ImageTk


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
        self.root.geometry('600x400')  # 調整窗口大小
        set_appearance_mode("dark")
        self.root.configure(bg='#1E1E1E')  # 更柔和的深色背景

        style = Style()
        style.theme_use("default")
        style.configure(
            "Yellow.Horizontal.TProgressbar",
            troughcolor="#1E1E1E",  # 黑色背景
            background="#FFCC70",  # 黃色進度
            thickness=5
        )

        # 標題區域
        self.title_label = CTkLabel(
            master=root,
            text="人臉辨識系統",
            font=("Helvetica", 24, "bold"),
            text_color="#FFCC70"
        )
        self.title_label.pack(pady=10)

        # 上傳按鈕
        self.upload_button = CTkButton(
            master=root,
            text="Upload",
            command=self.upload_image,
            corner_radius=20,
            fg_color="#FFCC70",
            hover_color="#FFA940",
            text_color="#1E1E1E",
            font=("Arial", 12, "bold"),
            width=200
        )
        self.upload_button.pack(pady=15)

        # 結果顯示區域
        self.result_label = CTkLabel(
            master=root,
            text="Please upload the picture",
            corner_radius=15,
            font=("Arial", 14),
            fg_color="#2C2C2C",
            text_color="#FFFFFF",
            width=400,
            height=80
        )
        self.result_label.pack(pady=20)

        self.progress = Progressbar(
            root,
            orient="horizontal",
            mode="determinate",
            length=400,
            style="Yellow.Horizontal.TProgressbar"
        )
        self.progress.pack(pady=10)
        self.progress.pack_forget()

    def upload_image(self):
        """用戶上傳圖片並執行人臉辨識"""
        file_path = filedialog.askopenfilename(
            title="選擇圖片",
            filetypes=[("圖片檔案", "*.jpg;*.jpeg;*.png")]
        )

        if file_path:

            try:
                # 假設 face_db.find_matching_face 是人臉匹配的函數
                self.progress.pack()
                self.progress["value"] = 0
                self.root.update_idletasks()

                # 模擬進度更新（實際處理中可同步更新進度）
                self.simulate_processing()

                match = self.face_db.find_matching_face(file_path)

                if match:
                    self.result_label.configure(
                        text=f"Matched Face: {match['name']}\nSimilarity: {match['similarity']:.4f}",
                        text_color="#A7E22E"
                    )
                else:
                    self.result_label.configure(
                        text="No matched face exist",
                        text_color="#FF5555"
                    )
            except Exception as e:
                messagebox.showerror("error", f"處理圖片時發生錯誤：{e}")

    def simulate_processing(self):
        """模擬進度條更新"""
        for i in range(1, 101, 1):  # 模擬分10步完成
            self.progress["value"] = i
            self.root.update_idletasks()
            self.root.after(30)


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
        root = CTk()
        app = FaceRecognitionApp(root, face_db)
        root.mainloop()
    finally:
        face_db.close_connection()


if __name__ == "__main__":
    main()





