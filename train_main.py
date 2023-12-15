from ultralytics import YOLO
 
# Load the model.
model = YOLO('yolov8n.pt')

model.train(
   data='data.yaml',
   imgsz=1000,
   batch=4,
   epochs=1000)
