import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import cv2 
                  
def visual(image,labels):
      # 定义不同器官对应的颜色映射
      colors = {
            0: (0, 0, 0),        # 背景 - 黑色
            1: (255, 0, 0),      # 器官 1 - 红色
            2: (0, 255, 0),      # 器官 2 - 绿色
            3: (0, 0, 255),      # 器官 3 - 蓝色
            4: (255, 255, 0),    # 器官 4 - 黄色
            5: (255, 0, 255),    # 器官 5 - 品红色
            6: (0, 255, 255),    # 器官 6 - 青色
            7: (128, 0, 0),      # 器官 7 - 深红色
            8: (0, 128, 0),      # 器官 8 - 深绿色
            9: (0, 0, 128),      # 器官 9 - 深蓝色
            10: (128, 128, 0),   # 器官 10 - 深黄色
            11: (128, 0, 128),   # 器官 11 - 深品红色
            12: (0, 128, 128),   # 器官 12 - 深青色
            13: (192, 192, 192)  # 器官 13 - 银色
      }
      
      # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4))
      
      # 创建彩色掩码
      mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.float32)

      # 将器官标签对应的像素位置设置为对应颜色
      for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                  label = labels[i, j]
                  if label != 0:
                        color = colors[label]
                        mask[i, j] = [color[0] / 255.0, color[1] / 255.0, color[2] / 255.0]  # 设置为对应颜色
                  else:
                        mask[i, j] = [image[i, j], image[i, j], image[i, j]]  # 保持原始黑白图像的样子

      # 可视化彩色掩码
      plt.imshow(mask)
      plt.axis('off')
      
      plt.savefig('pred_36_130.png')

def plot_legend():
      colors = {
            0: (0, 0, 0),        # 背景 - 黑色
            1: (255, 0, 0),      # 器官 1 - 红色
            2: (0, 255, 0),      # 器官 2 - 绿色
            3: (0, 0, 255),      # 器官 3 - 蓝色
            4: (255, 255, 0),    # 器官 4 - 黄色
            5: (255, 0, 255),    # 器官 5 - 品红色
            6: (0, 255, 255),    # 器官 6 - 青色
            7: (128, 0, 0),      # 器官 7 - 深红色
            8: (0, 128, 0),      # 器官 8 - 深绿色
            9: (0, 0, 128),      # 器官 9 - 深蓝色
            10: (128, 128, 0),   # 器官 10 - 深黄色
            11: (128, 0, 128),   # 器官 11 - 深品红色
            12: (0, 128, 128),   # 器官 12 - 深青色
            13: (192, 192, 192)  # 器官 13 - 银色
      }
      legend_elements = []
      for label, color in colors.items():
            if label != 0:
                  legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor=(color[0]/255.0,color[1]/255.0,color[2]/255.0), label=str(label)))

      plt.legend(handles=legend_elements)
      plt.axis('off')
      plt.tight_layout()
      plt.savefig('legend.png')
      
def visualimg(path) :
      image=nib.load(path)
      image = image.get_fdata()
      slice_data=image[:,:,130]
      image = Image.fromarray(slice_data)
      image = image.convert('RGB')
      image.save('img_36_130.png')

     
input_label=nib.load('../testresult/predictions/case0036_gt.nii')
input_pred=nib.load('../testresult/predictions/case0036_pred.nii')
input_image=nib.load('../testresult/predictions/case0036_img.nii')

input_pred = input_pred.get_fdata()
input_image = input_image.get_fdata()
input_label = input_label.get_fdata()



slice_data1=input_image[:,:,130]
slice_data2=input_pred[:,:,130]
slice_data3=input_label[:,:,130]

# visual(slice_data1,slice_data3)
visual(slice_data1,slice_data2)
# visualimg('/root/autodl-tmp/dataset/RawData/Training/img/img0036.nii')
# plot_legend()
