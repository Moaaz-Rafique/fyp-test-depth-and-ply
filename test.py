import numpy as np
import random
from PIL import Image

PLY_HEADER = 'ply\nformat ascii 1.0\nelement vertex {}\
  \nproperty float x\
  \nproperty float y\
  \nproperty float z\
  \nproperty uchar red\
  \nproperty uchar green\
  \nproperty uchar blue\
  \nend_header\n'


def write_ply_file(verts, faces, ply_file_name, write_also_npz=False):
    try:
        verts_num = verts.shape[0]
        faces_num = faces.shape[0]
        with open(ply_file_name, 'w') as f:
            f.write(PLY_HEADER.format(verts_num))
        with open(ply_file_name, 'ab') as f:
            for i in range(verts_num):
              f.write((f'{verts[i][0]} {verts[i][1]} {verts[i][2]} {faces[i][0]} {faces[i][1]} {faces[i][2]}\n').encode())
        return True
    except Exception as e:
        print('Error in write_ply_file! ({})'.format(ply_file_name))
        print(e)
        return False
# for i in range(10000):
#   vertices.append([random.random(), random.random(), random.random()])
for i_no in [35]:#,72, 5):
  vertices = []
  img = Image.open(f'output_images/{i_no}_depth.png')  
  col = Image.open(f'output_images/{i_no}_color.png')  
  image = np.array(img)
  col_img = np.array(col)
  # print(len(image), len(image[0]))
  print(image.shape)
  white_pixels=0
  colors=[]
  for i in range(len(image)):
    for j in range(len(image[0])):
      if not (col_img[i][j][0] == col_img[i][j][1] and col_img[i][j][0] == col_img[i][j][2] and col_img[i][j][0] == 255):
        white_pixels +=1
        vertices.append([i/1260.00 , j/945.00, image[i][j]/256.00])
        colors.append([col_img[i][j][0], col_img[i][j][1],col_img[i][j][2]])
  print(white_pixels, 1260*945)
  write_ply_file(np.array(vertices), np.array(colors), f'output_ply/pointcloud_{i_no}.ply')

  # print(np.array(vertices)[:10], np.array(colors)[:10])