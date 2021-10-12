from pathlib import Path
import cv2

## Pathlib explanations
## double asterisk (**/) means all subdirectories

pair_num = 1255

root_path =  Path(r"/mnt/data_disk/datasets/rellis-3d/Rellis-3D/") 

train_list_path = '/mnt/data_disk/datasets/rellis-3d/train.lst'

# 3302
img_list = [line.strip().split()[0] for line in open(train_list_path)]
# 3302
label_list = [line.strip().split()[1] for line in open(train_list_path)]

img_path = str( root_path / img_list[pair_num] )
label_path = str( root_path / label_list[pair_num] )

img = cv2.imread(img_path)
label = cv2.imread(label_path)

cv2.imshow("Image", img)


cv2.imshow("Label", label)
cv2.waitKey(0) 

#closing all open windows 
cv2.destroyAllWindows() 