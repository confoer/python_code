import torch
import os
import cv2
from PIL import Image
from model import EmotionNet
from config import config
from utils import get_transforms

class EmotionPredictor:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EmotionNet(num_classes=len(config.classes))
        self.model.load_state_dict(torch.load(model_path, map_location=self.device,weights_only=True))
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = get_transforms()[1]
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces

    def predict_emotion(self, face_image):
        img_tensor = self.transform(Image.fromarray(face_image)).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, preds = torch.max(probs, 1)
            
        return {
            "emotion": config.classes[preds.item()],
            "confidence": conf.item()
        }
        
    def predict(self, image):
        faces = self.detect_faces(image)
        results = []
        for (x, y, w, h) in faces:
            face_image = image[y:y+h, x:x+w]
            result = self.predict_emotion(face_image)
            results.append({
                "emotion": result["emotion"],
                "confidence": result["confidence"],
                "bbox": (x, y, w, h)
            })
        return results
    
def run_camera(predictor):
    cap = cv2.VideoCapture(0)  # 参数0表示打开默认摄像头
    cv2.namedWindow("Emotion Detection", cv2.WINDOW_NORMAL)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("无法读取图像帧！")
            break 
        
        predictions = predictor.predict(frame)
        
        for pred in predictions:
            x, y, w, h = pred["bbox"]
            emotion = pred["emotion"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            text = f"{emotion} "
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Emotion Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):#按下q键退出
            break

    cap.release()
    cv2.destroyAllWindows()
    return text

# def run_image_prediction(predictor, image_path, save_result=False, output_path=None, show_result=True):
#     image = cv2.imread(image_path)
#     if image is None:
#         raise FileNotFoundError(f"无法读取图像文件: {image_path}")
    
#     predictions = predictor.predict(image)

#     for pred in predictions:
#         x, y, w, h = pred["bbox"]
#         emotion = pred["emotion"]
        
#         cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#         text = f"{emotion}"
#         cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
#     annotated_image = image.copy()

#     if save_result:
#         if output_path is None:
#             output_path = "annotated_image.jpg"
#         cv2.imwrite(output_path, annotated_image)
#         print(f"标注后的图像已保存到: {output_path}")
    
#     if show_result:
#         cv2.namedWindow("Emotion Detection", cv2.WINDOW_NORMAL)
#         cv2.imshow("Emotion Detection", annotated_image)
#         cv2.waitKey(0)
#         cv2.destroyAllWindows()

#     return text
def run_image_prediction(predictor, image_path, save_result=False, output_path=None, show_result=True):
    image = cv2.imread(image_path)
    # if image is None:
    #     raise FileNotFoundError(f"无法读取图像文件: {image_path}")
    
    predictions = predictor.predict(image)

    for pred in predictions:
        x, y, w, h = pred["bbox"]
        emotion = pred["emotion"]
        
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = f"{emotion}"
        cv2.putText(image, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    annotated_image = image.copy()

    if save_result:
        if output_path is None:
            output_path = "annotated_image.jpg"
        cv2.imwrite(output_path, annotated_image)
        print(f"标注后的图像已保存到: {output_path}")
    
    if show_result:
        cv2.namedWindow("Emotion Detection", cv2.WINDOW_NORMAL)
        cv2.imshow("Emotion Detection", annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return text

if __name__ == '__main__':
    if not os.path.exists("Match_Project\models\\best_model.pth"):
        raise FileNotFoundError("请先训练模型！")
    predictor = EmotionPredictor("Match_Project\models\\best_model.pth")
    
    image_path = "D:\Datasets\data\processed\\test\\angry\\20220230_000746.jpg"
    
    text = run_image_prediction(
        predictor, 
        image_path, 
        save_result=False, 
        show_result=True
    )    
    print(text)