import random
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import torch
import torch.nn as nn
from PIL import Image, ImageOps, ImageTk
from torchvision import transforms


class ImprovedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(32 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = torch.flatten(x, start_dim=1)
        x = self.dropout(self.relu(self.fc1(x)))
        return self.fc2(x)


class QualityControlApp:
    class_names = ("LỖI (Defective)", "ĐẠT (OK)")

    def __init__(self, window):
        self.window = window
        self.repo_root = Path(__file__).resolve().parent
        self.dataset_root = (
            self.repo_root / "data" / "casting_data" / "casting_data"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.pipeline_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]),
        ])
        self.photo_images = []
        self._build_ui()

    def _load_model(self):
        model_path = self.repo_root / "final_casting_model.pth"
        if not model_path.is_file():
            raise FileNotFoundError(
                f"Không tìm thấy trọng số mô hình: {model_path}. "
                "Hãy chạy notebook 03_model_evaluation.ipynb trước."
            )

        model = ImprovedCNN().to(self.device)
        state_dict = torch.load(
            model_path,
            map_location=self.device,
            weights_only=True,
        )
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        return model

    def _build_ui(self):
        self.window.title("Kiểm soát chất lượng tự động")
        self.window.geometry("1120x760")
        self.window.minsize(850, 620)

        container = ttk.Frame(self.window, padding=18)
        container.pack(fill="both", expand=True)
        container.columnconfigure(0, weight=1, uniform="cards")
        container.columnconfigure(1, weight=1, uniform="cards")
        container.rowconfigure(3, weight=1)

        ttk.Label(
            container,
            text="KIỂM SOÁT CHẤT LƯỢNG TỰ ĐỘNG",
            font=("Segoe UI", 18, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(
            container,
            text=f"Mô hình CNN | Thiết bị: {self.device}",
            font=("Segoe UI", 10),
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 14))

        controls = ttk.Frame(container)
        controls.grid(row=2, column=0, columnspan=2, sticky="new", pady=(0, 14))
        ttk.Button(
            controls,
            text="Chọn ảnh để kiểm tra",
            command=self._choose_image,
        ).pack(side="left", padx=(0, 8))
        ttk.Button(
            controls,
            text="Chạy thử ngẫu nhiên: 1 lỗi + 1 đạt",
            command=self._run_random_pair,
        ).pack(side="left")
        self.status_label = ttk.Label(controls, text="Sẵn sàng kiểm tra")
        self.status_label.pack(side="right")

        self.cards = []
        for column in range(2):
            card = ttk.LabelFrame(
                container,
                text=f"Vật thể {column + 1}",
                padding=12,
            )
            card.grid(
                row=3,
                column=column,
                sticky="nsew",
                padx=(0, 8) if column == 0 else (8, 0),
            )
            image_label = ttk.Label(card, anchor="center")
            image_label.pack(fill="both", expand=True, pady=(0, 8))
            file_label = ttk.Label(
                card,
                text="Chưa có ảnh",
                anchor="center",
                wraplength=480,
            )
            file_label.pack(fill="x")
            result_label = tk.Label(
                card,
                text="",
                font=("Segoe UI", 15, "bold"),
                anchor="center",
            )
            result_label.pack(fill="x", pady=(8, 2))
            confidence_label = ttk.Label(card, text="", anchor="center")
            confidence_label.pack(fill="x")
            expected_label = ttk.Label(card, text="", anchor="center")
            expected_label.pack(fill="x", pady=(2, 0))
            self.cards.append({
                "frame": card,
                "image": image_label,
                "file": file_label,
                "result": result_label,
                "confidence": confidence_label,
                "expected": expected_label,
            })

        self._clear_cards()

    def _predict(self, image):
        input_tensor = self.pipeline_transform(image).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(input_tensor), dim=1)
            confidence, predicted_class = probabilities.max(dim=1)
        return predicted_class.item(), confidence.item() * 100

    def _choose_image(self):
        image_path = filedialog.askopenfilename(
            title="Chọn ảnh sản phẩm",
            filetypes=[
                ("Ảnh", "*.jpeg *.jpg *.png *.bmp"),
                ("Tất cả tệp", "*.*"),
            ],
        )
        if image_path:
            self._inspect_paths([(Path(image_path), None)])

    def _run_random_pair(self):
        defect_folder = self.dataset_root / "test" / "def_front"
        ok_folder = self.dataset_root / "test" / "ok_front"
        defect_images = self._find_images(defect_folder)
        ok_images = self._find_images(ok_folder)
        if not defect_images:
            messagebox.showerror(
                "Không tìm thấy ảnh lỗi",
                f"Không có ảnh trong thư mục:\n{defect_folder}",
                parent=self.window,
            )
            return
        if not ok_images:
            messagebox.showerror(
                "Không tìm thấy ảnh đạt",
                f"Không có ảnh trong thư mục:\n{ok_folder}",
                parent=self.window,
            )
            return

        self._inspect_paths([
            (random.choice(defect_images), 0),
            (random.choice(ok_images), 1),
        ])

    @staticmethod
    def _find_images(folder):
        if not folder.is_dir():
            return []
        return [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpeg", ".jpg", ".png", ".bmp"}
        ]

    def _inspect_paths(self, image_specs):
        self._clear_cards()
        for index, (image_path, expected_class) in enumerate(image_specs):
            try:
                with Image.open(image_path) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
            except OSError as error:
                messagebox.showerror(
                    "Không mở được ảnh",
                    f"{image_path}\n\n{error}",
                    parent=self.window,
                )
                self.status_label.configure(text="Không thể mở ảnh")
                return

            try:
                predicted_class, confidence = self._predict(image)
            except (RuntimeError, ValueError) as error:
                messagebox.showerror(
                    "Lỗi suy luận",
                    str(error),
                    parent=self.window,
                )
                self.status_label.configure(text="Suy luận thất bại")
                return

            self._show_result(
                index,
                image_path,
                image,
                predicted_class,
                confidence,
                expected_class,
            )

        self.status_label.configure(text=f"Đã kiểm tra {len(image_specs)} vật thể")

    def _show_result(
        self,
        index,
        image_path,
        image,
        predicted_class,
        confidence,
        expected_class,
    ):
        card = self.cards[index]
        preview = image.copy()
        preview.thumbnail((490, 430))
        photo = ImageTk.PhotoImage(preview)
        self.photo_images.append(photo)

        class_name = self.class_names[predicted_class]
        color = "#c62828" if predicted_class == 0 else "#218838"
        card["image"].configure(image=photo)
        card["file"].configure(text=image_path.name)
        card["result"].configure(
            text=f"Dự đoán: {class_name}",
            fg=color,
        )
        card["confidence"].configure(text=f"Độ tự tin: {confidence:.1f}%")
        if expected_class is not None:
            expected = self.class_names[expected_class]
            verdict = "ĐÚNG" if predicted_class == expected_class else "SAI"
            card["expected"].configure(text=f"Kỳ vọng: {expected} | {verdict}")
        card["frame"].grid()

    def _clear_cards(self):
        self.photo_images.clear()
        for card in self.cards:
            card["image"].configure(image="")
            card["file"].configure(text="Chưa có ảnh")
            card["result"].configure(text="")
            card["confidence"].configure(text="")
            card["expected"].configure(text="")
            card["frame"].grid_remove()


def main():
    window = tk.Tk()
    window.withdraw()
    try:
        QualityControlApp(window)
    except (FileNotFoundError, OSError, RuntimeError, ValueError, KeyError) as error:
        messagebox.showerror(
            "Không thể khởi động ứng dụng",
            str(error),
            parent=window,
        )
        window.destroy()
        return

    window.deiconify()
    window.mainloop()


if __name__ == "__main__":
    main()
