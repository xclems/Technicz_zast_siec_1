import tkinter as tk
from tkinter import messagebox
import numpy as np

# --- Математична частина (MLP з 2 прихованими шарами) ---
class DeepNeuralNetwork:
    def __init__(self):
        # Розміри шарів
        self.input_size = 784
        self.h1_size = 128
        self.h2_size = 64
        self.output_size = 3
        
        # Ініціалізація ваг (He initialization)
        self.W1 = np.random.randn(self.input_size, self.h1_size) * np.sqrt(2./self.input_size)
        self.b1 = np.zeros((1, self.h1_size))
        
        self.W2 = np.random.randn(self.h1_size, self.h2_size) * np.sqrt(2./self.h1_size)
        self.b2 = np.zeros((1, self.h2_size))
        
        self.W3 = np.random.randn(self.h2_size, self.output_size) * np.sqrt(2./self.h2_size)
        self.b3 = np.zeros((1, self.output_size))
        
        self.lr = 0.05 # Швидкість навчання

    def relu(self, x):
        return np.maximum(0, x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def forward(self, X):
        # Прохід вперед через 3 шари
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)
        
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.softmax(self.z3)
        return self.a3

    def train(self, X, y_label):
        target = np.zeros((1, 3))
        target[0, y_label] = 1
        
        # Forward
        output = self.forward(X)
        
        # Backpropagation (ланцюгове правило)
        # Шар 3 (вихідний)
        dz3 = output - target
        dW3 = np.dot(self.a2.T, dz3)
        db3 = np.sum(dz3, axis=0, keepdims=True)
        
        # Шар 2 (прихований 2)
        dz2 = np.dot(dz3, self.W3.T) * (self.z2 > 0)
        dW2 = np.dot(self.a1.T, dz2)
        db2 = np.sum(dz2, axis=0, keepdims=True)
        
        # Шар 1 (прихований 1)
        dz1 = np.dot(dz2, self.W2.T) * (self.z1 > 0)
        dW1 = np.dot(X.T, dz1)
        db1 = np.sum(dz1, axis=0, keepdims=True)
        
        # Оновлення всіх ваг
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W3 -= self.lr * dW3
        self.b3 -= self.lr * db3

# --- GUI Частина ---
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("MLP: P-R-O Recognizer")
        self.nn = DeepNeuralNetwork()
        
        self.canvas_size = 280
        self.grid_res = 28
        self.pixel_w = self.canvas_size // self.grid_res
        self.drawing_data = np.zeros((self.grid_res, self.grid_res))

        # Створення інтерфейсу
        self.setup_ui()

    def setup_ui(self):
        # Полотно ліворуч
        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size, bg="white", highlightthickness=1, highlightbackground="black")
        self.canvas.grid(row=0, column=0, rowspan=5, padx=20, pady=20)
        self.canvas.bind("<B1-Motion>", self.draw)

        # Кнопки літер праворуч
        self.btns = []
        letters = ["P", "R", "O"]
        for i, char in enumerate(letters):
            btn = tk.Button(self.root, text=char, width=12, font=('Arial', 12, 'bold'),
                            command=lambda idx=i: self.on_letter_btn_click(idx))
            btn.grid(row=i, column=1, padx=10, sticky="n")
            self.btns.append(btn)

        # Чекбокс навчання
        self.is_training = tk.BooleanVar()
        self.chk_train = tk.Checkbutton(self.root, text="Навчання", variable=self.is_training, font=('Arial', 10))
        self.chk_train.grid(row=3, column=1)

        # Нижні кнопки керування
        self.btn_done = tk.Button(self.root, text="Готово", bg="#e1e1e1", width=15, command=self.predict)
        self.btn_done.grid(row=5, column=0, pady=5)

        self.btn_reset = tk.Button(self.root, text="Ресет", bg="#ffcccc", width=15, command=self.clear_all)
        self.btn_reset.grid(row=5, column=1, pady=5)

    def draw(self, event):
        x, y = event.x // self.pixel_w, event.y // self.pixel_w
        if 0 <= x < self.grid_res and 0 <= y < self.grid_res:
            # Малюємо жирніше (зачіпаємо сусідні пікселі для кращого розпізнавання)
            for dx in range(2): 
                for dy in range(2):
                    if x+dx < self.grid_res and y+dy < self.grid_res:
                        self.drawing_data[y+dy, x+dx] = 1.0
                        self.canvas.create_rectangle((x+dx)*self.pixel_w, (y+dy)*self.pixel_w, 
                                                    (x+dx+1)*self.pixel_w, (y+dy+1)*self.pixel_w, 
                                                    fill="black", outline="")

    def on_letter_btn_click(self, idx):
        if self.is_training.get():
            input_vec = self.drawing_data.flatten().reshape(1, -1)
            if np.sum(input_vec) > 0:
                self.nn.train(input_vec, idx)
                self.clear_all()
        else:
            messagebox.showinfo("Підказка", "Ці кнопки для навчання. Увімкніть чекбокс 'Навчання'.")

    def predict(self):
        if self.is_training.get():
            messagebox.showwarning("Увага", "Вимкніть режим навчання для розпізнавання!")
            return
            
        input_vec = self.drawing_data.flatten().reshape(1, -1)
        if np.sum(input_vec) == 0: return
        
        outputs = self.nn.forward(input_vec)
        winner = np.argmax(outputs)
        
        self.reset_btn_colors()
        self.btns[winner].configure(bg="green", fg="white")

    def reset_btn_colors(self):
        for btn in self.btns:
            btn.configure(bg="#f0f0f0", fg="black")

    def clear_all(self):
        self.drawing_data.fill(0)
        self.canvas.delete("all")
        self.reset_btn_colors()

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()