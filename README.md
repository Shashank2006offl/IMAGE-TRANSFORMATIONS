# IMAGE-TRANSFORMATIONS
  
## Aim
To perform image transformation such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping using OpenCV and Python.

## Software Required:
Anaconda - Python 3.7

## Algorithm:
### Step1:
Import all the necessary modules  

### Step2:
Choose an image and save it as filename.jpg

### Step3:
Use imread to read the image

### Step4:
Use cv2.warpPerspective(image,M,(cols,rows)) to translation the image

### Step5:
Use cv2.warpPerspective(image,M,(cols2,rows2)) to scale the image

### Step6:
Use cv2.warpPerspective(image,M,(int(cols1.5),int(rows1.5))) for x and y axis to shear the image

### Step7:
Use cv2.warpPerspective(image,M,(int(cols),int(rows))) for x and y axis to reflect the image

### Step8:
Use cv2.warpPerspective(image,M,(int(cols),int(rows))) to rotate the image

### Step9:
Crop the image to remove unwanted areas from an image

### Step10:
Use cv2.imshow to show the image

### Step11:
End the program

## Program:

Developed By: SHASHANK R
Register Number: 212223230205

i)Image Translation
```
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
```

# Note that we only need the first two rows of the matrix for warpAffine
M_affine = M[:2, :]

translated_img = cv2.warpAffine(input_img, M_affine, (cols, rows))

plt.axis('off')
plt.imshow(translated_img)
plt.show()
```

ii) Image Scaling
```
scaled_img=cv2.warpPerspective(input_img,M,(cols,rows))
plt.axis('off')
plt.imshow(scaled_img)
plt.show()

iii)Image shearing
```
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
```

iv)Image Reflection
```
import numpy as np
import cv2
import matplotlib.pyplot as plt

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
```

v)Image Rotation
```
rotated_image = cv2.warpAffine(img, cv2.getRotationMatrix2D((5620//2, 3747//2), 45, 1.0), (5620, 3747))
  
plt.imshow(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))  
plt.title("Rotated Image") 
plt.axis('off')
```

vi)Image Cropping
```
import numpy as np
import cv2
import matplotlib.pyplot as plt

input_image = cv2.imread("/content/old-rusty-fishing-boat-slope-along-shore-lake.jpg")
input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
x, y, w, h = 100, 100, 300, 300
cropped_image = input_image[y:y+h, x:x+w]

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(input_image)
ax[0].set_title("Original Image")
ax[1].imshow(cropped_image)
ax[1].set_title("Cropped Image")
plt.show()
```


## Output:
### i)Image Translation
![image](https://github.com/user-attachments/assets/f8fab2e9-42e3-4be4-bb77-9059067657a6)

### ii) Image Scaling
![image](https://github.com/user-attachments/assets/50530052-1a8d-4e0e-8ec1-5ff5fc33c230)

### iii)Image shearing
![image](https://github.com/user-attachments/assets/51b1515c-85aa-4a0b-965d-63cd8d06b739)

### iv)Image Reflection
![image](https://github.com/user-attachments/assets/726d9411-dbe7-4626-8e33-aaec83f1cb6a)

### v)Image Rotation
![image](https://github.com/user-attachments/assets/24941095-428d-496b-a265-bd866bca4f7b)

### vi)Image Cropping
![image](https://github.com/user-attachments/assets/d6b44f78-a7ea-4de9-a462-020e2d1b7120)

## Result: 

Thus the different image transformations such as Translation, Scaling, Shearing, Reflection, Rotation and Cropping are done using OpenCV and python programming.
