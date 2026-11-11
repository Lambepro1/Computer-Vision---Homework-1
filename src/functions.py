import cv2
import os
import csv
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity as ssim
from PIL import Image, ImageDraw, ImageFont


#--------------------------------
# Function definitions
def metrics_computing(img1, img2, index, results):
    # PSNR
    # How close the pixels are to the original image
    # a good value should be above 30dB
    psnr_v = peak_signal_noise_ratio(img1, img2)
    print(f"PSNR of image {index} and its filtered version: {psnr_v:.2f} dB")

    # SSIM
    # How similar the image structure (edges, texture) looks to the human eye
    # Closer to 1 is better
    img_deg_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img_str_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    score = ssim(
                        img_deg_gray, 
                        img_str_gray, 
                        full=False, 
                        data_range=img_deg_gray.max() - img_deg_gray.min())
    results.append((psnr_v, score))
    print(f"SSIM Score of image {index} and its filtered version: {score}\n")   
    return 


def csv_builder(csv_file_path, results, method):

    with open(csv_file_path, mode='a', newline='') as file_csv:
        writer = csv.writer(file_csv, delimiter='|')
        
        writer.writerow([method])
        writer.writerow(['img', 'PSNR', 'SSIM'])
        
        for idx, (psnr, ssim) in enumerate(results): # write values for every image
            writer.writerow([idx, round(psnr, 4), round(ssim, 4)]) # round to 4 dec and print the results
        
        writer.writerow([]) # white row


def album_builder(albums, titles = None):
    if titles is None:
        titles = ['Original', 'Degraded', 'Linear Contrast Stretching', 'CLAHE', 'Bilateral Filter + saturation']

    padding = 20
    header_h = 60
    font = ImageFont.load_default(size=60)

    for i in albums:  # check if each album contains exactly 5 images
        num_img = len(i)
        if num_img != 5:
            raise ValueError("Each album must contain exactly 5 images.")

    for i in range(num_img):
        set_of_same_img = [album[i] for album in albums]  # extract the i-th image from each album

        h, w = set_of_same_img[0].shape[:2]
        for img in set_of_same_img:
            if img.shape[:2] != (h, w):  # check if all images have the same dimensions and 3 channels
                raise ValueError("All images must have the same dimensions.")

        total_width = (w * num_img) + (padding * (num_img + 1))
        total_height = header_h + h + (padding * 2)

        canvas = Image.new('RGB', (total_width, total_height), color='white')
        draw = ImageDraw.Draw(canvas)

        for col_idx, (title, img_arr) in enumerate(zip(titles, set_of_same_img)):
            x_offset = padding + col_idx * (w + padding)

            x_text = x_offset + int(w * 0.30)
            draw.text((x_text, 15), title, fill="black", font=font) # add the title centered

            img_clean = np.clip(img_arr, 0, 255).astype(np.uint8)
            
            # converts img array to PIL image
            if len(img_arr.shape) == 2: # check if img is 2D (grayscale) or 3D 
                img_pil = Image.fromarray(img_clean, mode='L')
            else:
                img_clean = img_clean[:, :, ::-1]
                img_pil = Image.fromarray(img_clean)
                
            canvas.paste(img_pil, (x_offset, header_h + padding))

        output_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(output_dir, exist_ok=True)

        filename = f"img{i}_results.png"
        filepath = os.path.join(output_dir, filename)
        canvas.save(filepath)

        print(f"Saved: {filename} ({h}x{w})")
    return
