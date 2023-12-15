# Defect Inspection On Motherboard
## Description
Defect inspection is one of hot topic in computer vision, deep learning for smart manufacturing. Here, we implemented a webapp system for defect inspection on motherboard to defect possible flaws. The base model is a pretrained YOLOv8n customized to make prediction on 11 distinct labels.

## Requirements
 - Ultralytics
 - Streamlit
 - PIL

## Dataset
The dataset used for training is original from roboflow. It includes 1015 images annotated in YOLOv8 format with the following classes: CPU_FAN_NO_Screws, CPU_FAN_Screw_loose, CPU_FAN_Screws, CPU_fan, CPU_fan_port, CPU_fan_port_detached, Incorrect_Screws, Loose_Screws, No_Screws, Scratch, Screws.

The following pre-processing was applied to each image:
* Auto-orientation of pixel data (with EXIF-orientation stripping)
* Resize to 1000x1000 (Stretch)
* Auto-contrast via adaptive equalization
  
The following augmentation was applied to create 3 versions of each source image:
* 50% probability of horizontal flip
* Equal probability of one of the following 90-degree rotations: none, clockwise, counter-clockwise
* Random exposure adjustment of between -25 and +25 percent

The data ratio is as follow:
* Training set: 939 images
* Validation set: 31 images
* Test set: 45 images

![dataset](https://github.com/WENDGOUNDI/defect_inspection_on_motherboard/assets/48753146/f8a412bf-e534-40f8-9b7a-809457ea665c)

## Training
The model has been launched to train for 1000 epochs with patience set to 50 and the training was completed after 115 epochs. The network is based on a pretrained YOLOv8n. mAP@50 was 0.88.

## Deployment
The webapp has been developed via Streamlit. An interactive webapp where the user can upload motherboard images with the followwing extension "jpg", "jpeg", "png", 'bmp', "webp" then adjust the confidence score ranging from 25 to 100. The inspect button after being pressed perform the inspection, side by side image between the original image and the predicted image. We also list the detected labels. The system is run locally at this moment.

![webapp](https://github.com/WENDGOUNDI/defect_inspection_on_motherboard/assets/48753146/5dca33ab-5bba-423c-8150-2ea0de46b0f0)

## How To Run
* For training, you may create a new virtual environment with this command **python -m venv my_yolo_env** . Once the virtual environment ready, open and pip install the required libraries mentioned above. In **data.yaml**, adjust the train, val and test path based on your directory. Open the training file **train_main.py** to select the desired pretrained yolov8 model you would like to use, adjust **data.yaml** file path as well as the number of epoch and other necessary parameters and run **python train_main.py**.
* Run **python inference.py** for inference. Adjust the testing image path based on your directory.
* For inference via the web app, in your virtual environment, run **streamlit run inspection_webapp.py**
