import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np

class ImageVarianceMeanCalculator:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Std Ratio Calculator")

        self.frame = tk.Frame(root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(self.frame, cursor="cross")
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.scroll_x = tk.Scrollbar(self.frame, orient=tk.HORIZONTAL, command=self.canvas.xview)
        self.scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

        self.scroll_y = tk.Scrollbar(self.frame, orient=tk.VERTICAL, command=self.canvas.yview)
        self.scroll_y.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.configure(xscrollcommand=self.scroll_x.set, yscrollcommand=self.scroll_y.set)

        self.button_frame = tk.Frame(root)
        self.button_frame.pack()

        self.load_button = tk.Button(self.button_frame, text="Load Image", command=self.load_image)
        self.load_button.pack(side=tk.LEFT)

        self.reset_button = tk.Button(self.button_frame, text="Reset", command=self.reset_rectangles)
        self.reset_button.pack(side=tk.LEFT)

        self.rectangles = []
        self.start_x = None
        self.start_y = None

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.original_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            self.display_image = cv2.cvtColor(self.original_image, cv2.COLOR_GRAY2RGB)
            self.image = Image.fromarray(self.display_image)
            self.photo = ImageTk.PhotoImage(self.image)
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
            self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))

    def on_button_press(self, event):
        self.start_x = self.canvas.canvasx(event.x)
        self.start_y = self.canvas.canvasy(event.y)
        self.rect_id = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='red')

    def on_mouse_drag(self, event):
        cur_x, cur_y = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        self.canvas.coords(self.rect_id, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        end_x, end_y = (self.canvas.canvasx(event.x), self.canvas.canvasy(event.y))
        self.rectangles.append((self.start_x, self.start_y, end_x, end_y))
        if len(self.rectangles) == 7:
            self.calculate_statistics()

    def calculate_statistics(self):
        std_values = []

        for i, rect in enumerate(self.rectangles):
            x1, y1, x2, y2 = rect
            x1, x2 = sorted([int(x1), int(x2)])
            y1, y2 = sorted([int(y1), int(y2)])
            roi = self.original_image[y1:y2, x1:x2]

            if roi.size == 0:
                print(f"Rectangle {i+1} is empty or out of bounds.")
                return

            std = np.std(roi)
            std_values.append(std)
            print(f"Std of rectangle {i+1}: {std:.4f}")

        ref_std = std_values[0]

        print("\nRatio of std (20*log10):")
        for i in range(1, 7):
            if std_values[i] == 0:
                ratio_db = float('inf')
            else:
                ratio = ref_std / std_values[i]
                ratio_db = 20 * np.log10(ratio)

            print(f"Rect {i+1} vs Rect 1: {ratio_db:.2f} dB")

    def reset_rectangles(self):
        self.rectangles.clear()
        self.canvas.delete("all")
        if hasattr(self, 'photo'):
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageVarianceMeanCalculator(root)
    root.mainloop()
