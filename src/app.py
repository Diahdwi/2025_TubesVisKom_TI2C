import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO
import os

# =========================
# Load YOLO Model
# =========================
model = YOLO("model/best.pt")

# =========================
# Window Setup
# =========================
root = tk.Tk()
root.title("Deteksi Kendaraan - YOLOv8")
root.geometry("1000x600")

# =========================
# Frames
# =========================
frame_img = tk.Frame(root)
frame_img.pack(pady=10)

frame_btn = tk.Frame(root)
frame_btn.pack(pady=15)

# =========================
# Labels
# =========================
label_before = tk.Label(frame_img, text="Before Detection")
label_before.grid(row=0, column=0, padx=10)

label_after = tk.Label(frame_img, text="After Detection")
label_after.grid(row=0, column=1, padx=10)

img_before = tk.Label(frame_img)
img_before.grid(row=1, column=0)

img_after = tk.Label(frame_img)
img_after.grid(row=1, column=1)

status = tk.Label(root, text="Silakan pilih gambar atau video", fg="blue")
status.pack()

# =========================
# Functions
# =========================
def pilih_file():
    file_path = filedialog.askopenfilename(
        filetypes=[
            ("Image", "*.jpg *.png *.jpeg"),
            ("Video", "*.mp4 *.avi *.mov")
        ]
    )

    if not file_path:
        return

    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".jpg", ".png", ".jpeg"]:
        status.config(text="Processing image...", fg="orange")
        root.update()
        proses_gambar(file_path)

    elif ext in [".mp4", ".avi", ".mov"]:
        status.config(text="Processing video... (tekan Q untuk keluar)", fg="orange")
        root.update()
        proses_video(file_path)


def proses_gambar(path):
    img = cv2.imread(path)
    if img is None:
        status.config(text="Gagal membaca gambar", fg="red")
        return

    # Before
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resize = cv2.resize(img_rgb, (400, 300))
    img_tk = ImageTk.PhotoImage(Image.fromarray(img_resize))
    img_before.config(image=img_tk)
    img_before.image = img_tk

    # YOLO Detection
    results = model.predict(path, conf=0.4, verbose=False)
    annotated = results[0].plot()

    annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    annotated = cv2.resize(annotated, (400, 300))
    img_tk2 = ImageTk.PhotoImage(Image.fromarray(annotated))
    img_after.config(image=img_tk2)
    img_after.image = img_tk2

    status.config(text="Deteksi gambar selesai", fg="green")


def proses_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        status.config(text="Gagal membuka video", fg="red")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=0.4, verbose=False)
        annotated = results[0].plot()

        cv2.imshow("Deteksi Kendaraan - Video (Tekan Q untuk keluar)", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    status.config(text="Deteksi video selesai", fg="green")


def reset():
    img_before.config(image="")
    img_after.config(image="")
    status.config(text="Silakan pilih gambar atau video", fg="blue")
    cv2.destroyAllWindows()

# =========================
# Buttons
# =========================
btn_pilih = tk.Button(frame_btn, text="Pilih File", width=15, command=pilih_file)
btn_pilih.grid(row=0, column=0, padx=10)

btn_reset = tk.Button(frame_btn, text="Deteksi Lagi", width=15, command=reset)
btn_reset.grid(row=0, column=1, padx=10)

btn_exit = tk.Button(frame_btn, text="Selesai", width=15, command=root.destroy)
btn_exit.grid(row=0, column=2, padx=10)

# =========================
# Run App
# =========================
root.mainloop()
