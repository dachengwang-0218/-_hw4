import os
import shutil
import tempfile

import cv2
import joblib
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_recall_fscore_support,
)
from torchvision import models, transforms


DEMO_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(DEMO_DIR)
TEST_DIR = os.path.join(BASE_DIR, "dataset", "test")
TASK_FILE = os.path.join(DEMO_DIR, "hand_landmarker.task")

SVM_MODEL_PATH = os.path.join(DEMO_DIR, "rps_svm_model.pkl")
MLP_MODEL_PATH = os.path.join(DEMO_DIR, "rps_mlp_model.pkl")
MOBILENET_MODEL_PATH = os.path.join(DEMO_DIR, "rps_mobilenetv3_model.pth")

LABEL_MAP = {"rock": 0, "paper": 1, "scissors": 2}
TARGET_NAMES = ["Rock", "Paper", "Scissors"]
IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def read_image(img_path):
    img_bytes = np.fromfile(img_path, dtype=np.uint8)
    if img_bytes.size == 0:
        return None
    return cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)


def collect_test_images():
    samples = []
    for category, label_idx in LABEL_MAP.items():
        category_dir = os.path.join(TEST_DIR, category)
        if not os.path.exists(category_dir):
            print(f"Warning: missing category folder: {category_dir}")
            continue

        for filename in os.listdir(category_dir):
            if filename.lower().endswith(IMAGE_EXTS):
                samples.append((os.path.join(category_dir, filename), label_idx))

    return samples


def make_svm_features(samples):
    x_data, y_data = [], []
    for img_path, label_idx in samples:
        img = read_image(img_path)
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))
        x_data.append(resized.flatten())
        y_data.append(label_idx)

    return np.array(x_data) / 255.0, np.array(y_data)


def prepare_task_file():
    temp_dir = tempfile.mkdtemp(prefix="aiot_hw4_")
    temp_task_file = os.path.join(temp_dir, "hand_landmarker.task")
    shutil.copy2(TASK_FILE, temp_task_file)
    return temp_task_file, temp_dir


def make_landmarker(task_file):
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=task_file),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def extract_landmarks(bgr_img, landmarker):
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_img)
    result = landmarker.detect(mp_img)
    if not result.hand_landmarks:
        return None

    landmarks = result.hand_landmarks[0]
    return np.array([[point.x, point.y, point.z] for point in landmarks]).flatten()


def make_mlp_features(samples):
    if not os.path.exists(TASK_FILE):
        raise FileNotFoundError(f"Missing MediaPipe task file: {TASK_FILE}")

    task_file, temp_dir = prepare_task_file()
    landmarker = make_landmarker(task_file)
    try:
        x_data, y_data = [], []
        for img_path, label_idx in samples:
            img = read_image(img_path)
            if img is None:
                continue
            features = extract_landmarks(img, landmarker)
            if features is None:
                continue
            x_data.append(features)
            y_data.append(label_idx)
    finally:
        landmarker.close()
        shutil.rmtree(temp_dir, ignore_errors=True)

    return np.array(x_data), np.array(y_data)


def load_mobilenet_model(device):
    checkpoint = torch.load(MOBILENET_MODEL_PATH, map_location=device)
    classes = checkpoint["classes"]

    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    return model, classes


def predict_mobilenet(samples):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, classes = load_mobilenet_model(device)
    class_to_common_idx = {name: LABEL_MAP[name] for name in classes}
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    y_true, y_pred = [], []
    with torch.no_grad():
        for img_path, label_idx in samples:
            img = read_image(img_path)
            if img is None:
                continue

            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)
            tensor = transform(pil_img).unsqueeze(0).to(device)
            pred_idx = int(model(tensor).argmax(1).item())
            pred_name = classes[pred_idx]

            y_true.append(label_idx)
            y_pred.append(class_to_common_idx[pred_name])

    return np.array(y_true), np.array(y_pred)


def summarize_result(model_name, y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        average="macro",
        zero_division=0,
    )
    return {
        "model": model_name,
        "samples": len(y_true),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def print_summary(results):
    print("\n=== Summary (macro average) ===")
    print(f"{'Model':<18} {'Samples':>8} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-score':>10}")
    print("-" * 80)
    for row in sorted(results, key=lambda item: item["accuracy"], reverse=True):
        print(
            f"{row['model']:<18} "
            f"{row['samples']:>8} "
            f"{row['accuracy'] * 100:>9.2f}% "
            f"{row['precision'] * 100:>9.2f}% "
            f"{row['recall'] * 100:>9.2f}% "
            f"{row['f1'] * 100:>9.2f}%"
        )


def evaluate_sklearn_model(model_name, model_path, x_test, y_test):
    if not os.path.exists(model_path):
        print(f"Skipping {model_name}: model not found: {model_path}")
        return None

    clf = joblib.load(model_path)
    y_pred = clf.predict(x_test)
    print(f"\n=== {model_name} ===")
    print(classification_report(y_test, y_pred, target_names=TARGET_NAMES, digits=4))
    return summarize_result(model_name, y_test, y_pred)


def main():
    if not os.path.exists(TEST_DIR):
        raise FileNotFoundError(f"Missing test dataset: {TEST_DIR}")

    samples = collect_test_images()
    if not samples:
        print("No test images found.")
        return

    print(f"Found {len(samples)} test images.")
    results = []

    print("\nPreparing SVM image features...")
    x_svm, y_svm = make_svm_features(samples)
    svm_result = evaluate_sklearn_model("SVM", SVM_MODEL_PATH, x_svm, y_svm)
    if svm_result:
        results.append(svm_result)

    print("\nPreparing MLP MediaPipe landmark features...")
    x_mlp, y_mlp = make_mlp_features(samples)
    print(f"MediaPipe usable samples: {len(y_mlp)} / {len(samples)}")
    mlp_result = evaluate_sklearn_model("MLP", MLP_MODEL_PATH, x_mlp, y_mlp)
    if mlp_result:
        results.append(mlp_result)

    if os.path.exists(MOBILENET_MODEL_PATH):
        print("\nRunning MobileNetV3 image predictions...")
        y_mobile_true, y_mobile_pred = predict_mobilenet(samples)
        print("\n=== MobileNetV3 ===")
        print(
            classification_report(
                y_mobile_true,
                y_mobile_pred,
                target_names=TARGET_NAMES,
                digits=4,
            )
        )
        results.append(summarize_result("MobileNetV3", y_mobile_true, y_mobile_pred))
    else:
        print(f"Skipping MobileNetV3: model not found: {MOBILENET_MODEL_PATH}")

    if results:
        print_summary(results)


if __name__ == "__main__":
    main()
