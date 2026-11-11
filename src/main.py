import cv2
import os
import numpy as np
from functions import metrics_computing, csv_builder, album_builder

#------------------------------------------------

if __name__ == "__main__":

    csv_file_path = "results.csv"
    with open(csv_file_path, mode='w', newline='') as f: # creation of the csv where the results will be saved
        pass

    # creation of distorted images
    img0 = cv2.imread('samples/lc_pd.jpg')
    img1 = cv2.imread('samples/bar.jpg')
    img2 = cv2.imread('samples/pd1.jpg')
    img3 = cv2.imread('samples/mtty.jpg')
    img4 = cv2.imread('samples/altare.jpg')

    album = [img0, img1, img2, img3, img4]
    album = [cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) for img in album]
    imgs_deg = []

    for img in album:
        img_blur = cv2.GaussianBlur(img, (15, 15), 0) # adding Gaussian blur to the image
        img_degraded = cv2.convertScaleAbs(img_blur, alpha=0.5, beta=25) # decreasing contrast
        imgs_deg.append(img_degraded)

    #-------------------------------------------------------
    # 2. Linear contrast stretching
    # channel stretching, applying min-max normalization to each channel
    album_stretched = []
    results1 = []

    print("\n------Metrics for Linear Contrast Stretching method:------\n")
    for j in range(len(imgs_deg)):
        album_stretched.append(np.zeros_like(imgs_deg[j]))
        for i in range(3):  # iterating on every channel
            channel = imgs_deg[j][:, :, i]
            album_stretched[j][:, :, i] = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX)
        
        partial_view = cv2.hconcat([imgs_deg[j], album_stretched[j]])
        cv2.imshow(f'Linear contrast stretching - image {j}', partial_view)

        metrics_computing(album[j], album_stretched[j], j, results1)
    csv_builder(csv_file_path, results1, "Linear Contrast Stretching")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #-------------------------------------------------------
    # 3. Histogram equalization
    # CLAHE method 
    clahe_imgs = []
    results2 = []

    print("\n------Metrics for CLAHE method:------\n")
    for i in range(len(imgs_deg)):
        clahe_img = cv2.cvtColor(imgs_deg[i], cv2.COLOR_BGR2HSV) # converting to hsv

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) # divison of the image in 8x8 square 
        clahe_img[:, :, 2] = clahe.apply(clahe_img[:, :, 2]) # application of CLAHE only on the v channel
        clahe_img_equalized = cv2.cvtColor(clahe_img, cv2.COLOR_HSV2BGR) # Convert back to RGB color space

        clahe_imgs.append(clahe_img_equalized)
        
        clahe_view = cv2.hconcat([imgs_deg[i], clahe_img_equalized])
        cv2.imshow(f'CLAHE of img {i}', clahe_view)

        metrics_computing(album[i], clahe_img_equalized, i, results2)
    csv_builder(csv_file_path, results2, "CLAHE")

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    #-------------------------------------------------------
    # 4. Fourth method
    # bilateral filter
    album_enhanced = []
    results3 = []

    print("\n------Metrics for Bilateral Filter method:------\n")
    for i in range(len(imgs_deg)):
        img_filtered = cv2.bilateralFilter(imgs_deg[i], 10, 80, 80)  # application of bilateral filter, it tends to smooths edges
        img_filtered_hsv = cv2.cvtColor(img_filtered, cv2.COLOR_BGR2HSV) 
        
        h, s, v = cv2.split(img_filtered_hsv) # to restore colours, I will raise the saturation channel
        s = np.clip(s * 1.20, 0, 255).astype(np.uint8) # to force to 0 or 255 pixels that will tend to overflow

        img_filtered_hsv = cv2.merge((h, s, v))
        img_enhanced = cv2.cvtColor(img_filtered_hsv, cv2.COLOR_HSV2BGR)
        album_enhanced.append(img_enhanced)

        bilateral_view = cv2.hconcat([imgs_deg[i], album_enhanced[i]])
        cv2.imshow(f'bilateral filter of img {i}', bilateral_view)
        
        metrics_computing(album[i], album_enhanced[i], i, results3)
    csv_builder(csv_file_path, results3, "Bilateral filter + saturation")

    cv2.waitKey(0)
    cv2.destroyAllWindows()


    albums = [album, imgs_deg, album_stretched, clahe_imgs, album_enhanced]
    album_builder(albums)
    print("\n\n Done!")