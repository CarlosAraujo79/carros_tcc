import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

# Carregar o modelo
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best_license_plate_model.pt', force_reload=True)

def detect_plate_and_text(image_path):
    results = model(image_path)
    labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    text_detected = ""

    for i, (label, cord) in enumerate(zip(labels, cords)):
        if label == 0:  # classe 0 (assumindo que seja 'placa')
            x1, y1, x2, y2 = cord[0], cord[1], cord[2], cord[3]
            w, h = img.size
            box = [x1 * w, y1 * h, x2 * w, y2 * h]
            draw.rectangle(box, outline="red", width=3)
            cropped = img.crop(box)
            
            # OCR
            import pytesseract
            plate_text = pytesseract.image_to_string(cropped, config='--psm 7')
            text_detected = plate_text.strip()

    return img, text_detected
