from ultralytics import YOLO

def finetune():
    model = YOLO("yolov8n.pt")

    model.train(
        data="/Users/patrickmartin/PycharmProjects/BetterDining/dataset/data.yaml",
        epochs=24,
        batch=8,
        imgsz=(854, 480),
        device="mps",
    )

    metrics = model.val()
    print(metrics)

if __name__ == "__main__":
    finetune()