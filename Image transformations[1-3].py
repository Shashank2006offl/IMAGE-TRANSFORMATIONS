
##i)Image Translation
import numpy as np
import cv2
import matplotlib.pyplot as plt

input_img = cv2.imread("/content/old-rusty-fishing-boat-slope-along-shore-lake.jpg")
input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

plt.axis('off')
plt.imshow(input_img)
plt.show()

rows, cols, dim = input_img.shape
M = np.float32([[1, 0, 2000], [0, 1, 50], [0, 0, 1]])



##ii)Image Scaling

scaled_img=cv2.warpPerspective(input_img,M,(cols,rows))
plt.axis('off')
plt.imshow(scaled_img)
plt.show()


##iii)Image Shearing
M_x = np.float32([[1, 0.2, 0], [0, 1, 0], [0, 0, 1]])
M_y = np.float32([[1, 0, 0], [0.2, 1, 0], [0, 0, 1]])

sheared_img_xaxis = cv2.warpPerspective(input_img, M_x, (cols, rows))
sheared_img_yaxis = cv2.warpPerspective(input_img, M_y, (cols, rows))

plt.axis('off')
plt.imshow(sheared_img_xaxis)
plt.show()

plt.axis('off')
plt.imshow(sheared_img_yaxis)
plt.show()

import numpy as np
import cv2
import matplotlib.pyplot as plt
##iv)Image Reflection
input_image = cv2.imread("/content/old-rusty-fishing-boat-slope-along-shore-lake.jpg")
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
plt.axis("off")
plt.imshow(input_image)
plt.show()
rows, cols, dim = input_image.shape
M_x = np.float32([[1, 0, 0], [0, -1, rows], [0, 0, 1]])
M_y = np.float32([[-1, 0, cols], [0, 1, 0], [0, 0, 1]])
reflected_img_xaxis = cv2.warpPerspective(input_image, M_x, (cols, rows))
reflected_img_yaxis = cv2.warpPerspective(input_image, M_y, (cols, rows))
plt.imshow(reflected_img_xaxis)  
plt.axis("off")
plt.show()

