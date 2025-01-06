import os
import cv2
import torch
import glob
import numpy as np
from PIL import Image
from torchvision import transforms
from model import BreastCancerModel
import argparse


# Định nghĩa các đối số đầu vào qua argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Breast Cancer Segmentation Prediction")
    
    parser.add_argument('--ckpt_path', type=str, required=True, help="Path to the checkpoint file")
    parser.add_argument('--save_dir', type=str, required=True, help="Directory to save the prediction results")
    parser.add_argument('--arch', type=str, default='UNET', choices=['UNET', 'UnetPlusPlus', 'MAnet', 'Linknet', 'FPN', 'PSPNet', 'DeepLabV3', 'DeepLabV3Plus', 'PAN', 'UPerNet', 'Segformer'], help="Architecture of the model")
    parser.add_argument('--encoder_name', type=str, default='resnet34', choices=['resnet34', 'mobilenet_v2', 'efficientnet-b7'], help="Encoder type")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image for prediction")

    return parser.parse_args()


def main():
    # Parse arguments
    args = parse_args()
    
    # Load checkpoint
    checkpoint = torch.load(args.ckpt_path, weights_only=True)
    
    model = BreastCancerModel(args.arch, args.encoder_name, in_channels=3, out_classes=1)
    model.load_state_dict(checkpoint['state_dict'])
    
    model.eval()

    # Load and preprocess the image
    image = cv2.cvtColor(cv2.imread(args.image_path), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = Image.fromarray(image) 

    transform = transforms.Compose([transforms.ToTensor()])
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(image_tensor)
        pr_mask = logits.sigmoid().numpy().squeeze()

    # Determine the current version number
    current_version = 0
    lst_version_name = [version_path.split('/')[-1] for version_path in glob.glob(os.path.join(args.save_dir, '*'))]
    for version in lst_version_name:
        if current_version < int(version):
            current_version = int(version)
    current_version += 1
    
    # Create a new directory to save the results
    save_path = os.path.join(args.save_dir, f"{current_version}")
    os.makedirs(save_path, exist_ok=True)

    # Save the image and mask
    image = np.array(image)
    cv2.imwrite(os.path.join(save_path, 'image.png'), image)

    pr_mask = (pr_mask * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_path, 'mask.png'), pr_mask)


if __name__ == '__main__':
    main()
