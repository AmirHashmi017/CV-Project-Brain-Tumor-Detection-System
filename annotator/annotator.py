import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import glob

CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]

class AnnotatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Medical Image Annotator")
        self.root.geometry("1000x700")

        self.image_folder = ""
        self.save_folder = "annotations"
        self.image_paths = []
        self.current_index = 0
        self.current_image = None
        self.display_image = None
        self.tk_image = None

        # Drawing state
        self.annotations = []  # List of dicts: {'type': 'bbox' or 'polygon', 'class_id': int, 'points': list}
        self.drawing = False
        self.current_points = []
        
        # Tool mode: 'bbox' or 'polygon'
        self.mode = tk.StringVar(value='bbox')
        self.selected_class = tk.IntVar(value=0)

        self.setup_ui()
        
    def setup_ui(self):
        # Sidebar
        sidebar = tk.Frame(self.root, width=200, bg="#f0f0f0", padx=10, pady=10)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        
        tk.Button(sidebar, text="Select Dataset Folder", command=self.load_folder).pack(fill=tk.X, pady=5)
        
        self.lbl_info = tk.Label(sidebar, text="0 / 0 images", bg="#f0f0f0")
        self.lbl_info.pack(pady=5)
        
        tk.Button(sidebar, text="Previous (P)", command=self.prev_image).pack(fill=tk.X, pady=2)
        tk.Button(sidebar, text="Next (N)", command=self.next_image).pack(fill=tk.X, pady=2)
        tk.Button(sidebar, text="Save (S)", command=self.save_annotations, bg="#d9ead3").pack(fill=tk.X, pady=10)
        tk.Button(sidebar, text="Delete Last", command=self.delete_last).pack(fill=tk.X, pady=2)
        tk.Button(sidebar, text="Clear All", command=self.clear_all).pack(fill=tk.X, pady=2)
        
        tk.Label(sidebar, text="Annotation Mode:", bg="#f0f0f0", font=("Arial", 10, "bold")).pack(pady=(15, 0))
        tk.Radiobutton(sidebar, text="Bounding Box", variable=self.mode, value='bbox', bg="#f0f0f0").pack(anchor=tk.W)
        tk.Radiobutton(sidebar, text="Polygon (Seg)", variable=self.mode, value='polygon', bg="#f0f0f0").pack(anchor=tk.W)

        tk.Label(sidebar, text="Classes:", bg="#f0f0f0", font=("Arial", 10, "bold")).pack(pady=(15, 0))
        for i, cls in enumerate(CLASSES):
            tk.Radiobutton(sidebar, text=f"{cls} ({i+1})", variable=self.selected_class, value=i, bg="#f0f0f0").pack(anchor=tk.W)

        # Canvas
        self.canvas = tk.Canvas(self.root, bg="gray", cursor="cross")
        self.canvas.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Binds
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<ButtonPress-3>", self.on_right_click) # For finishing polygon
        
        self.root.bind('<n>', lambda e: self.next_image())
        self.root.bind('<p>', lambda e: self.prev_image())
        self.root.bind('<s>', lambda e: self.save_annotations())
        for i in range(len(CLASSES)):
            self.root.bind(str(i+1), lambda e, idx=i: self.selected_class.set(idx))
            
    def load_folder(self):
        folder = filedialog.askdirectory(title="Select Image Folder")
        if not folder: return
        self.image_folder = folder
        
        # Save in the global annotations folder
        base_ann_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "annotations"))
        self.save_folder = os.path.join(base_ann_dir, "labels")
        self.visual_folder = os.path.join(base_ann_dir, "images")
        os.makedirs(self.save_folder, exist_ok=True)
        os.makedirs(self.visual_folder, exist_ok=True)
        
        self.image_paths = sorted(glob.glob(os.path.join(self.image_folder, "*.[pj][np][ge]*")))
        self.current_index = 0
        if self.image_paths:
            self.load_image()
        else:
            messagebox.showinfo("Info", "No images found in the selected folder.")
            
    def load_image(self):
        if not self.image_paths: return
        path = self.image_paths[self.current_index]
        self.current_image = cv2.imread(path)
        if self.current_image is None: return
        self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
        
        self.lbl_info.config(text=f"{self.current_index + 1} / {len(self.image_paths)}\n{os.path.basename(path)}")
        self.annotations = []
        self.current_points = []
        self.load_existing_annotations(path)
        self.update_display()
        
    def load_existing_annotations(self, img_path):
        filename = os.path.basename(img_path)
        label_filename = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(self.save_folder, label_filename)
        
        if not os.path.exists(label_path): return
        
        h, w = self.current_image.shape[:2]
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if not parts: continue
                class_id = int(parts[0])
                
                if len(parts) == 5: # BBox
                    cx, cy, bw, bh = map(float, parts[1:])
                    x1 = int((cx - bw/2) * w)
                    y1 = int((cy - bh/2) * h)
                    x2 = int((cx + bw/2) * w)
                    y2 = int((cy + bh/2) * h)
                    self.annotations.append({'type': 'bbox', 'class_id': class_id, 'points': [(x1,y1), (x2,y2)]})
                elif len(parts) > 5: # Polygon
                    points = []
                    for i in range(1, len(parts), 2):
                        px = int(float(parts[i]) * w)
                        py = int(float(parts[i+1]) * h)
                        points.append((px, py))
                    self.annotations.append({'type': 'polygon', 'class_id': class_id, 'points': points})
                    
    def update_display(self):
        if self.current_image is None: return
        display = self.current_image.copy()
        
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        
        for ann in self.annotations:
            color = colors[ann['class_id'] % len(colors)]
            if ann['type'] == 'bbox':
                pt1, pt2 = ann['points']
                cv2.rectangle(display, pt1, pt2, color, 2)
                cv2.putText(display, CLASSES[ann['class_id']], (pt1[0], pt1[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            elif ann['type'] == 'polygon':
                pts = [pt for pt in ann['points']]
                import numpy as np
                cv2.polylines(display, [np.array(pts)], isClosed=True, color=color, thickness=2)
                if pts:
                    cv2.putText(display, CLASSES[ann['class_id']], (pts[0][0], pts[0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw current in progress
        color = colors[self.selected_class.get() % len(colors)]
        if self.mode.get() == 'bbox' and len(self.current_points) == 2:
            cv2.rectangle(display, self.current_points[0], self.current_points[1], color, 2)
        elif self.mode.get() == 'polygon' and len(self.current_points) > 0:
            import numpy as np
            if len(self.current_points) > 1:
                cv2.polylines(display, [np.array(self.current_points)], isClosed=False, color=color, thickness=2)
            for pt in self.current_points:
                cv2.circle(display, pt, 3, color, -1)

        self.display_image = Image.fromarray(display)
        self.tk_image = ImageTk.PhotoImage(image=self.display_image)
        
        self.canvas.delete("all")
        # Center image on canvas
        c_w = self.canvas.winfo_width()
        c_h = self.canvas.winfo_height()
        if c_w <= 1: c_w = 800
        if c_h <= 1: c_h = 700
        
        i_w = self.display_image.width
        i_h = self.display_image.height
        
        self.img_x = max(0, (c_w - i_w) // 2)
        self.img_y = max(0, (c_h - i_h) // 2)
        
        self.canvas.create_image(self.img_x, self.img_y, anchor=tk.NW, image=self.tk_image)
        
    def _get_image_coords(self, event):
        x = event.x - self.img_x
        y = event.y - self.img_y
        if self.current_image is not None:
            x = max(0, min(x, self.current_image.shape[1] - 1))
            y = max(0, min(y, self.current_image.shape[0] - 1))
        return (x, y)

    def on_press(self, event):
        if self.current_image is None: return
        x, y = self._get_image_coords(event)
        
        if self.mode.get() == 'bbox':
            self.drawing = True
            self.current_points = [(x, y), (x, y)]
        elif self.mode.get() == 'polygon':
            self.current_points.append((x, y))
            self.update_display()
            
    def on_drag(self, event):
        if not self.drawing or self.mode.get() != 'bbox': return
        x, y = self._get_image_coords(event)
        self.current_points[1] = (x, y)
        self.update_display()
        
    def on_release(self, event):
        if not self.drawing or self.mode.get() != 'bbox': return
        self.drawing = False
        x, y = self._get_image_coords(event)
        self.current_points[1] = (x, y)
        
        if abs(self.current_points[0][0] - self.current_points[1][0]) > 5 and \
           abs(self.current_points[0][1] - self.current_points[1][1]) > 5:
            self.annotations.append({
                'type': 'bbox',
                'class_id': self.selected_class.get(),
                'points': list(self.current_points)
            })
        self.current_points = []
        self.update_display()

    def on_right_click(self, event):
        if self.mode.get() == 'polygon' and len(self.current_points) > 2:
            self.annotations.append({
                'type': 'polygon',
                'class_id': self.selected_class.get(),
                'points': list(self.current_points)
            })
            self.current_points = []
            self.update_display()
            
    def save_annotations(self):
        if not self.image_paths or self.current_image is None: return
        path = self.image_paths[self.current_index]
        filename = os.path.basename(path)
        label_filename = os.path.splitext(filename)[0] + ".txt"
        label_path = os.path.join(self.save_folder, label_filename)
        
        h, w = self.current_image.shape[:2]
        
        with open(label_path, "w") as f:
            for ann in self.annotations:
                cls_id = ann['class_id']
                if ann['type'] == 'bbox':
                    x1, y1 = ann['points'][0]
                    x2, y2 = ann['points'][1]
                    xmin, xmax = min(x1, x2), max(x1, x2)
                    ymin, ymax = min(y1, y2), max(y1, y2)
                    
                    cx = ((xmin + xmax) / 2) / w
                    cy = ((ymin + ymax) / 2) / h
                    bw = (xmax - xmin) / w
                    bh = (ymax - ymin) / h
                    
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")
                elif ann['type'] == 'polygon':
                    points_str = " ".join([f"{px/w:.6f} {py/h:.6f}" for px, py in ann['points']])
                    f.write(f"{cls_id} {points_str}\n")
                    
        # Save visual image
        display = self.current_image.copy()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
        for ann in self.annotations:
            color = colors[ann['class_id'] % len(colors)]
            if ann['type'] == 'bbox':
                pt1, pt2 = ann['points']
                cv2.rectangle(display, pt1, pt2, color, 2)
                cv2.putText(display, CLASSES[ann['class_id']], (pt1[0], pt1[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            elif ann['type'] == 'polygon':
                pts = [pt for pt in ann['points']]
                import numpy as np
                cv2.polylines(display, [np.array(pts)], isClosed=True, color=color, thickness=2)
                if pts:
                    cv2.putText(display, CLASSES[ann['class_id']], (pts[0][0], pts[0][1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        display_bgr = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
        visual_path = os.path.join(self.visual_folder, filename)
        cv2.imwrite(visual_path, display_bgr)
        
        print(f"Saved {len(self.annotations)} annotations to {label_path} and visual to {visual_path}")
        
    def delete_last(self):
        if self.annotations:
            self.annotations.pop()
            self.update_display()
            
    def clear_all(self):
        self.annotations = []
        self.current_points = []
        self.update_display()

    def next_image(self):
        if self.image_paths and self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.load_image()

    def prev_image(self):
        if self.image_paths and self.current_index > 0:
            self.current_index -= 1
            self.load_image()

if __name__ == "__main__":
    root = tk.Tk()
    app = AnnotatorApp(root)
    # Ensure window adjusts correctly on start
    root.update()
    app.update_display()
    root.mainloop()