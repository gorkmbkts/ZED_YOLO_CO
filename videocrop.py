import cv2
import numpy as np
# Görüntüyü yükleyin
image = cv2.imread("img3.png")

# Sol üst ve sağ alt köşelerin koordinatları
x1, y1 = 975, 390
x2, y2 = 1040, 455

# Görüntüyü crop edin
cropped_image = image[y1:y2, x1:x2]

dim=(640,640)
            
imgz = cv2.resize(cropped_image, dim, interpolation=cv2.INTER_CUBIC) #INTER_CUBIC interpolation
# Keskinleştirme çekirdeği tanımlayın

# Cropped görüntüyü gösterin
cv2.imshow("Cropped Image", imgz)
cv2.waitKey(0)
cv2.destroyAllWindows()
