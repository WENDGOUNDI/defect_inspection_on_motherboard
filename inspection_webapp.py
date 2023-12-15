import PIL
import streamlit as st
from ultralytics import YOLO

# Give the path of the best.pt (best weights)
model_path = 'best.pt'

# Setting page layout
st.set_page_config(
    page_title="MOTHERBOARD DEFECT INSPECTION WEBAPP SYSTEM)",  # Setting page title
    page_icon="image path to be the webapp icon",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded",    # Expanding sidebar by default
    
)

# Creating sidebar
with st.sidebar:
    st.header("Image Config")     # Adding header to sidebar
    # Adding file uploader to sidebar for selecting images
    source_img = st.file_uploader(
        "Upload an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))

    # Model Options
    confidence = float(st.slider(
        "Select Model Confidence", 25, 100, 40)) / 100

# Creating main page heading
st.title("MOTHERBOARD DEFECT INSPECTION WEBAPP SYSTEM")
st.caption('Updload a photo by selecting :blue[Browse files]')
st.caption('Then click the :blue[Detect Objects] button and check the result.')
# Creating two columns on the main page
col1, col2 = st.columns(2)

# Adding image to the first column if image is uploaded
with col1:
    if source_img:
        # Opening the uploaded image
        uploaded_image = PIL.Image.open(source_img)
        image_width, image_height = (644, 644) #uploaded_image.size
        # Adding the uploaded image to the page with a caption
        st.image(source_img,
                 caption="Uploaded Image",
                 width=image_width
                 )

try:
    model = YOLO(model_path)
except Exception as ex:
    st.error(
        f"Unable to load model. Check the specified path: {model_path}")
    st.error(ex)

if st.sidebar.button('INSPECT'):
    results = model.predict(uploaded_image,
                        conf=confidence,
                        line_width=3, 
                        show_labels=True, 
                        show_conf=False
                        )
    boxes = results[0].boxes
    results_plotted = results[0].plot(labels=True, line_width=4)[:, :, ::-1]
    with col2:
        st.image(results_plotted,
                 caption='Detected Image',
                 width=image_width                 
                 )
        try:
            names = model.names
            st.write(f'Number of detected objects: {len(boxes)}')
            predicted_label = list()
            for r in results:
                for c in r.boxes.cls:
                    predicted_label = names[int(c)]
                    st.text(predicted_label)
                      
        except Exception as ex:
            st.write("No image is uploaded yet!")
