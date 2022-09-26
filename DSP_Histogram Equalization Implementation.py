import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img_filename = 'C:/Users/Busra/Desktop/BIL561_Odev1/test2.jpg'
# loading image
image = cv.imread(img_filename, 0)
(h, w) = image.shape[:2]

#make image array
image_ar=np.asarray(image)

#find number of pixels of original image
hist,bins = np.histogram(image_ar, 256, [0,256])

#PDF calculation
pdf_of_image=hist/hist.sum()

#CDF calculation
#first, cumulative summution (Sk),cdf
sum=np.cumsum(pdf_of_image)
#second, max grey level value (sk*255) round the values for finding histogram equalization
hel= np.round(255*sum).astype(np.uint8)


# Transformation to image 
# array to list
image_list = list(image_ar)
# pixel values
hep_image_list = [hel[p] for p in image_list]
# reshape to histogram equalised picture
output_1 = np.reshape(np.asarray(hep_image_list), image.shape)


#histogram equalization with openCV
output_2 = cv.equalizeHist(image)

hist2, bins2 = np.histogram(output_1, 256, [0,256])
hist3, bins3 = np.histogram(output_2, 256, [0,256])

#difference between output_1 and output_2
difference= np.abs(output_1 - output_2)


#SECOND QUESTION
#modified histogram equalization (MHE) function
# reload image
img = cv.imread(img_filename, 0)
(h, w) = img.shape[:2]

# convert to array
img_array2 = np.asarray(img)

# flatten image array and calculate histogram
histogram_array2 = np.bincount(img_array2.flatten(), minlength=256)

#cumulative histogram(cdf)
chistogram_array2 = np.cumsum(histogram_array2)

#finding minimum element from cdf but should not be equal 0
minNonZero = chistogram_array2[np.min(np.nonzero(chistogram_array2))]
chistogram_array2 = chistogram_array2 - minNonZero

#MHE output
T = np.round((255.0 / ((img.shape[0] * img.shape[1]) - minNonZero)) * chistogram_array2).astype(np.uint8)

# Transformation to image 
# array to list
img_list2 = list(img_array2.flatten())
# pixel values
eq_img_list2 = [T[p] for p in img_list2]

# reshape to histogram equalised picture
output_3 = np.reshape(np.asarray(eq_img_list2), img_array2.shape)


hist4, bins4 = np.histogram(output_3.ravel(), 256, [0,256])


#OUTPUTS
#PLOTS
#plot number of pixels of original image
plt.subplot(221)
plt.plot(hist)
plt.title('Number of Pixels of original image')
plt.xticks([0,256])

#plot histogram equalization output
plt.subplot(222)
plt.plot(hist2)
plt.title('Number of Pixels of histogram equalization output')
plt.xticks([0,256])

#plot histogram equalization with opencv
plt.subplot(223)
plt.plot(hist3)
plt.title('Number of Pixels of histogram equalization with opencv output')
plt.xticks([0,256])

#plot modified histogram equalization 
plt.subplot(224)
plt.plot(hist4)
plt.title('Number of Pixels of modified histogram equalization output')
plt.xticks([0,256])



#IMAGES
#show original image
cv.imshow('Orginal image', image)
#show histogram equalization image
cv.imshow('Output 1', output_1)
#show histogram equalization image with opencv
cv.imshow('Output 2', output_2)
#difference between output1 and output2
cv.imshow('Absolute Difference (output_1 - output_2)', 100*difference)
#show modified histogram equalization image
cv.imshow('Output 3', output_3)
#difference between output1 and output3
cv.imshow('Absolute Difference (output_1 - output_3)', 100*(np.abs(output_1 - output_3)))
#difference between output2 and output3
cv.imshow('Absolute Difference (output_2 - output_3)', 100*(np.abs(output_2 - output_3)))

#TERMINAL OUTPUTS
print("------------------------------------------")
print("Difference (output_1 - output_2) is : ", np.sum(np.sum(difference)))
print("------------------------------------------")
print("Difference (output_1 - output_3) is : ", np.sum(np.sum(np.abs(output_1 - output_3))))
print("------------------------------------------")
print("Difference (output_2 - output_3) is : ", np.sum(np.sum(np.abs(output_2 - output_3))))


plt.show()
cv.waitKey(0)
cv.destroyAllWindows()

#Results
#the difference between output1 and output2 is found 484
#the difference between output1 and output3 is found 484 
#the difference between output2 and output3 is found 0 





