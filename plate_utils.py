from ultralytics import YOLO
from PIL import Image, ImageDraw
import pytesseract

# Carrega o modelo YOLO (funciona sem OpenGL)
model = YOLO("best_license_plate_model.pt")

def detect_plate_and_text(image_path):
    # Faz a predição
    results = model(image_path)
    boxes = results[0].boxes.xyxy.cpu().numpy()  # coordenadas das bounding boxes

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    text_detected = ""

    for box in boxes:
        x1, y1, x2, y2 = box[:4]
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        # Recorta e aplica OCR
        cropped = img.crop([x1, y1, x2, y2])
        plate_text = pytesseract.image_to_string(cropped, config="--psm 7")
        if plate_text.strip():
            text_detected = plate_text.strip()

    return img, text_detected
