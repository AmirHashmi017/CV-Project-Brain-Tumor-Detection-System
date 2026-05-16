import os
import json
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory, url_for

app = Flask(__name__)

# Configuration
IMAGE_DIR = os.path.join(os.getcwd(), "temp_data", "images")
LABEL_DIR = os.path.join(os.getcwd(), "annotations", "labels")
VISUAL_DIR = os.path.join(os.getcwd(), "annotations", "images")
os.makedirs(IMAGE_DIR, exist_ok=True)
os.makedirs(LABEL_DIR, exist_ok=True)
os.makedirs(VISUAL_DIR, exist_ok=True)

DEFAULT_CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]

@app.route('/')
def index():
    return render_template('index.html', classes=DEFAULT_CLASSES)

@app.route('/api/images')
def get_images():
    if not os.path.exists(IMAGE_DIR):
        return jsonify([])
    files = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))]
    return jsonify(sorted(files))

@app.route('/api/upload', methods=['POST'])
def upload_images():
    if 'files' not in request.files:
        return jsonify({"success": False, "error": "No files provided"})
    
    files = request.files.getlist('files')
    uploaded = []
    for file in files:
        if file.filename:
            path = os.path.join(IMAGE_DIR, file.filename)
            file.save(path)
            uploaded.append(file.filename)
    
    return jsonify({"success": True, "uploaded": uploaded})

@app.route('/api/load_annotations/<filename>')
def load_annotations(filename):
    label_file = os.path.join(LABEL_DIR, os.path.splitext(filename)[0] + ".txt")
    img_path = os.path.join(IMAGE_DIR, filename)
    img = cv2.imread(img_path)
    if img is None:
        return jsonify({"error": "Image not found"}), 404
    
    h, w = img.shape[:2]
    anns = []
    if os.path.exists(label_file):
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                cls_id = int(parts[0])
                if len(parts) == 5:
                    cx, cy, bw, bh = map(float, parts[1:])
                    x1 = int((cx - bw/2)*w); y1 = int((cy - bh/2)*h)
                    x2 = int((cx + bw/2)*w); y2 = int((cy + bh/2)*h)
                    anns.append({"type": "bbox", "class_id": cls_id, "points": [[x1,y1],[x2,y2]]})
                elif len(parts) > 5:
                    pts = [[int(float(parts[i])*w), int(float(parts[i+1])*h)] for i in range(1, len(parts), 2)]
                    anns.append({"type": "polygon", "class_id": cls_id, "points": pts})
    return jsonify({"annotations": anns, "width": w, "height": h})

@app.route('/api/save', methods=['POST'])
def save_annotations():
    data = request.json
    filename = data.get('filename')
    anns = data.get('annotations', [])
    
    img_path = os.path.join(IMAGE_DIR, filename)
    img = cv2.imread(img_path)
    if img is None:
        return jsonify({"success": False, "error": "Image not found"})
    
    h, w = img.shape[:2]
    
    # Save YOLO labels
    label_path = os.path.join(LABEL_DIR, os.path.splitext(filename)[0] + ".txt")
    with open(label_path, "w") as f:
        for ann in anns:
            cls_id = ann["class_id"]
            if ann["type"] == "bbox":
                p1, p2 = ann["points"]
                xmin, xmax = min(p1[0], p2[0]), max(p1[0], p2[0])
                ymin, ymax = min(p1[1], p2[1]), max(p1[1], p2[1])
                cx = ((xmin+xmax)/2)/w; cy = ((ymin+ymax)/2)/h
                bw = (xmax-xmin)/w; bh = (ymax-ymin)/h
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
            elif ann["type"] == "polygon":
                pts_str = " ".join([f"{px/w:.6f} {py/h:.6f}" for px, py in ann["points"]])
                f.write(f"{cls_id} {pts_str}\n")
                
    # Save visual verification image
    display = img.copy()
    colors = [(220,50,50),(50,200,80),(50,100,230),(230,200,40),(180,50,220),(50,220,230),(230,120,50),(230,50,130)]
    for ann in anns:
        color = colors[ann["class_id"] % len(colors)]
        if ann["type"] == "bbox":
            p1, p2 = tuple(ann["points"][0]), tuple(ann["points"][1])
            cv2.rectangle(display, p1, p2, color[::-1], 2)
        elif ann["type"] == "polygon":
            pts_arr = np.array(ann["points"])
            cv2.polylines(display, [pts_arr], isClosed=True, color=color[::-1], thickness=2)
            
    cv2.imwrite(os.path.join(VISUAL_DIR, filename), display)
    return jsonify({"success": True})

@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(IMAGE_DIR, filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
