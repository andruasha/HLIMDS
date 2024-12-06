import time
import torch
import numpy as np
from torchvision import models, transforms
import cv2
import json

torch.backends.quantized.engine = 'qnnpack'

cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 224)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 224)
cap.set(cv2.CAP_PROP_FPS, 36)

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

weights = models.quantization.MobileNet_V2_QuantizedWeights.IMAGENET1K_QNNPACK_V1
net = models.quantization.mobilenet_v2(weights=weights, quantize=True)
net.eval()

with open('imagenet_class_index.json', 'r') as f:
    class_labels = json.load(f)
frame_count = 0
last_logged = time.time()

with torch.no_grad():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Не удалось получить кадр.")
            break

        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        input_tensor = preprocess(rgb_image).unsqueeze(0)

        output = net(input_tensor)

        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, class_index = torch.max(probabilities, 0)
        class_name = class_labels[str(class_index.item())][1]  

        text = f"{class_name}: {confidence.item():.2f}"
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("MobileNet Object Detection", frame)

        frame_count += 1
        now = time.time()
        if now - last_logged > 1:
            print(f"{frame_count / (now - last_logged):.2f} fps")
            last_logged = now
            frame_count = 0

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
