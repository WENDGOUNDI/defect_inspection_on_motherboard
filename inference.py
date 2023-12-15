# Import Libraries

from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

# Define testing image
image = "path to your testing image"
# Set the weight
model = YOLO("best.pt")
# Start prediction on the testing image
results = model.predict(source=image,conf=0.5, save=True) # save=True allow us to save the predicted image
# Retrieved training labels
names = model.names

# Loop to display predicted labels
for r in results:
    for c in r.boxes.cls:
        predicted_label = names[int(c)]
        print(f" Predicted Labels are : {predicted_label}")
