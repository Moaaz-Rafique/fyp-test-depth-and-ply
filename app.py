import torch
from PIL import Image
import requests
import numpy as np
import matplotlib.pyplot as plt
# from google.colab import files

from os import listdir
# import open3d as o3d
import io


def loadImages(path):
    # return array of images

    imagesList = listdir(path)
    loadedImages = []
    for image in imagesList:
        img = Image.open(path + image)
        loadedImages.append(img)

    return loadedImages




midas_type = "DPT_Large"

model = torch.hub.load("intel-isl/MiDaS", midas_type)
gpu_device = torch.device('cpu')
model.to(gpu_device)
model.eval()

transform = torch.hub.load('intel-isl/MiDaS', 'transforms').dpt_transform

def estimate_depth(image):
    transformed_image = transform(image).to(gpu_device)
    
    with torch.no_grad():
        prediction = model(transformed_image)
        
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    
    output = prediction.cpu().numpy()
    return output

path = "test_images/"

selected_images = loadImages(path)
for i in range(len(selected_images)):
  image = np.array(selected_images[i])
  output = estimate_depth(image)
  print(f"Output of image {i} generated")
  Image.fromarray(image.astype('uint8'), 'RGB').save(f'output_images/{i}_color.png')
  Image.fromarray(output.astype('uint8'), 'L').save(f'output_images/{i}_depth.png')
  
#   img = o3d.io.read_image(f"output_images/color{i}.png")
#   depth = o3d.io.read_image(f"output_images/depth{i}.png")
#   rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(img, depth)
#   pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
#       rgbd_image,
#       o3d.camera.PinholeCameraIntrinsic(
#           o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault))
#   # Flip it, otherwise the pointcloud will be upside down
#   pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
#   o3d.io.write_point_cloud(f"point_cloud{i+2}.ply", 
#                           pcd,
#                           write_ascii=True, 
#                           compressed=False, 
#                           print_progress=True
#                           )

# print(len(output))


