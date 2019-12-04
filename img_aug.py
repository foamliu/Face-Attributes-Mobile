# import the OpenCV package
import cv2

# load the image with imread()
imageSource = 'images/0_img.jpg'
img = cv2.imread(imageSource)

# copy image to display all 4 variations
horizontal_img = img.copy()
vertical_img = img.copy()
both_img = img.copy()

# flip img horizontally, vertically,
# and both axes with flip()
horizontal_img = cv2.flip(img, 1)
vertical_img = cv2.flip(img, 0)
both_img = cv2.flip(img, -1)

# display the images on screen with imshow()
cv2.imshow("Original", img)
cv2.imshow("Horizontal flip", horizontal_img)
cv2.imshow("Vertical flip", vertical_img)
cv2.imshow("Both flip", both_img)

# wait time in milliseconds
# this is required to show the image
# 0 = wait indefinitely
cv2.waitKey(0)

# close the windows
cv2.destroyAllWindows()