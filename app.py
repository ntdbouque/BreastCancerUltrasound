import gradio as gr
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from src.model import BreastCancerModel

import pydicom
import pandas as pd
import numpy as np



import os
os.environ["GRADIO_TEMP_DIR"] = "/workspace/competitions/Sly/CV_Final_Final/temp_gradio"


clahe = cv2.createCLAHE(
    clipLimit = 2., 
    tileGridSize = (10, 10)
)


checkpoint = torch.load('/workspace/competitions/Sly/CV_Final_Final/experiment/TensorBoardLogger/version_1/checkpoints/epoch=49-step=350.ckpt', weights_only=True)
    
model = BreastCancerModel('UNET', 'resnet34', in_channels=3, out_classes=1)
model.load_state_dict(checkpoint['state_dict'])

model.eval()


def read_dicom(path):
    # Read the DICOM file
    dicom = pydicom.dcmread(path)
    image = dicom.pixel_array

    # Convert to a numpy array of type float32
    image = image.astype(np.float32)

    # Convert to a PyTorch tensor
    image = torch.from_numpy(image)

    # Reshape if it's a single channel image
    if len(image.shape) == 2:
        image = image.unsqueeze(0)  # Add a channel dimension

    # Resize the image to 500x500
    resize_transform = transforms.Resize((500, 500))
    image = resize_transform(image)

    # Normalize the image
    image = image - torch.min(image)
    image = image / torch.max(image)
    image = image * 255.0

    # Convert to uint8
    image = image.to(torch.uint8)
    # Change shape from (1, 500, 500) to (500, 500, 1)
    image = image.squeeze(0).unsqueeze(2)
    return image

def enhance(fpath):

    image = read_dicom(fpath)

    img_clahe = clahe.apply(image.numpy()) 
    
    img_origin = image.squeeze().numpy()
    
    img_he = cv2.equalizeHist(image.numpy())
    
    return img_origin, img_he, img_clahe

def segment(image):
    image = cv2.resize(image, (224, 224))
    image = Image.fromarray(image) 
    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(image_tensor)
        pr_mask = logits.sigmoid().numpy().squeeze()

    return (pr_mask * 255).astype(np.uint8)

with gr.Blocks() as demo:
    title = gr.HTML("<h1><center>🚀 Breast Cancer Ultrasound Image Application</center></h1>")
    with gr.Tab("Breast Cancer Segmentation"):
        gr.Markdown("""
        🔍  Phân đoạn vùng bị ảnh hưởng bởi ung thư vú trên hình ảnh siêu âm.  Tải lên hình ảnh siêu âm và nhận kết quả phân đoạn tự động với mô hình Unet.
        """)
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(value='/workspace/competitions/Sly/CV_Final_Final/data/train/benign/benign (1).png')
                button_predict = gr.Button('Predict')           
            output_image = gr.Image(value='/workspace/competitions/Sly/CV_Final_Final/data/train/benign/benign (1)_mask.png')
            
    with gr.Tab("Image Enhancement"):
        gr.Markdown("""
        🎨  Cải thiện chất lượng hình ảnh y tế thông qua các kỹ thuật tăng cường như Histogram Equalization (HE) và CLAHE. Tải lên hình ảnh DICOM và xem kết quả cải thiện.
        """)
        with gr.Row():
            with gr.Column():
                input_image_enhance = gr.File(value='/workspace/competitions/Sly/CV_Final_Final/data/dicom/004f33259ee4aef671c2b95d54e4be68.dicom')
                button_enhance = gr.Button('Enhance')
                
            with gr.Row():
                output_image_origin = gr.Image(label='Original Image')
                output_image_he = gr.Image(label='HE Image')
                output_image_enhance = gr.Image(label='CLAHE Image')
            
    button_predict.click(segment, inputs=input_image, outputs=output_image)
    
    button_enhance.click(enhance, 
                         inputs=input_image_enhance, 
                         outputs=[output_image_origin, output_image_he, output_image_enhance])
    
demo.launch(debug=True, share=True)