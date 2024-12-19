from utils import compare_faces
from train import train_model

if __name__ == "__main__":
    # 設置數據路徑
    data_folder = "D:/Desktop/Pic/data/"
    save_dir = "face_recognition_models"

    try:
        # 訓練模型
        # embedding_model, history = train_model(data_folder, save_dir)
        # print("模型訓練完成！")

        # 測試模型
        test_image1 = "D:/Desktop/Pic/data/anchor05.jpg"
        test_image2 = "D:/Desktop/Pic/data/anchor07.jpg"

        # similarity = compare_faces(test_image1, test_image2, embedding_model)
        #if similarity is not None:
            #print(f"人臉相似度: {similarity:.4f}")
            #print(f"相似度閾值建議: 0.7 (大於此值可能是同一個人)")

        facenet_similarity = facenet_compare_faces(test_image1, test_image2)
        if facenet_similarity is not None:
            print(f"facenet的人臉相似度: {facenet_similarity:.4f}")
            print(f"相似度閾值建議: 0.7 (大於此值可能是同一個人)")

    except Exception as e:
        print(f"發生錯誤: {str(e)}")
