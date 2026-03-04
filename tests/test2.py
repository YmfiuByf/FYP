import torch
import transformers
import numpy as np
from teager import Teager
# from .teager import horizontal_teager
# from .teager import vertical_teager
# from .teager import diagonal_teager_right
# from .teager import diagonal_teager_left
# from .teager import crop_center


#model = torch.load("E:\\Edge download\\w2v_large_lv_fsh_swbd_cv_ftls960_updated.pt")
test_array = np.array([[1,2,3],[4,9,6],[7,8,9],[4,6,2],[7,4,9],[100,4,30]])
test_array2 = np.array([[1,2,3],[4,9,6],[7,8,9],[1,1,2],[7,4,9],[100,4,0]])
new_array = Teager(test_array,'horizontal',1)
new_array2 = Teager(test_array2,'horizontal',1)
print(new_array-new_array2)

