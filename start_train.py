import os

from roboflow import Roboflow

from ultralytics import YOLO

rf = Roboflow(api_key="EwKEYueIj4aRlz8EomTZ")
project = rf.workspace("s-workspace-rp148").project("alphabet-uit8m-rzbhb")
version = project.version(1)
dataset = version.download("yolov8")

model = YOLO("yolov8s.pt")
model.info()
results = model.train(
    data=os.path.join(dataset.location, "data.yaml"),
    epochs=150,
    imgsz=640,
    batch=16,
    name="alphabet_200_model(s(100))",
    device="cuda:0",
    save_period=10,
)
print("completed training")
