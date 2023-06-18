import numpy as np
import nibabel as nib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

pred=nib.load('../testresult/predictions/case0036_pred.nii')
pred = pred.get_fdata()

volume_shape = pred.shape
volume_data = np.zeros(volume_shape)

class_to_name = {1: 'spleen', 2: 'right kidney', 3: 'left kidney', 4: 'gallbladder', 5: ' esophagus', 6: ' liver', 7: 'stomach', 8: 'aorta'
                 ,9:'inferior vena cava', 10: 'portal vein and splenic vein', 11:'pancreas', 
                 12:'right adrenal gland', 13:'left adrenal gland'}
# 配置器官颜色
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

print('start')

for organ_num in range(1,14):     
      fig = plt.figure()
      ax = fig.add_subplot(111, projection='3d')
      
      color=colors[organ_num]

      indices=np.where(pred==organ_num)
      volume_data[indices] = True
      ax.scatter(indices[0], indices[1], indices[2], c=(color[0]/255.0, color[1]/255.0, color[2]/255.0), marker='o')

      ax.set_xlabel('X')
      ax.set_ylabel('Y')
      ax.set_zlabel('Z')
      ax.grid(False)
      
      path='volumn/pred_36_'+class_to_name[organ_num]+'.png'
      plt.savefig(path)
      print(organ_num)