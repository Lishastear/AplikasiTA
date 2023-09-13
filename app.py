# app.py - This is the main Flask application file.

# Import the Flask module
from flask import Flask, request, render_template, jsonify, url_for
from PIL import Image
import pywt
import numpy as np
import cv2
import shutil
import os
from skimage import metrics
from skimage.metrics import mean_squared_error, structural_similarity
from scipy.fftpack import dct
from scipy.fftpack import idct
from math import log10, sqrt
from werkzeug.utils import secure_filename

# Create a Flask web application
app = Flask(__name__)
# app.static_folder = 'static'
app.config['UPLOAD_FOLDER'] = 'uploads'  # Anda dapat mengganti 'uploads' dengan nama folder yang Anda inginkan

# source_file = 'uploads/file.png'
# destination_file = 'static/file.png'

# shutil.move(source_file, destination_file)
#DCT-DWT
def convert_image(image_name, size):
    img = Image.open('./' + image_name).resize((size, size), 1)
    img = img.convert('L')
 
    image_array = np.array(img.getdata(), dtype=np.float64).reshape((size, size))   

    return image_array

def process_coefficients(imArray, model, level):
    coeffs=pywt.wavedec2(data = imArray, wavelet = model, level = level)
    coeffs_H=list(coeffs) 
   
    return coeffs_H
            
    
def embed_watermark(watermark_array, orig_image):
    watermark_array_size = watermark_array[0].__len__()
    watermark_flat = watermark_array.ravel()
    ind = 0

    for x in range (0, orig_image.__len__(), 8):
        for y in range (0, orig_image.__len__(), 8):
            if ind < watermark_flat.__len__():
                subdct = orig_image[x:x+8, y:y+8]
                subdct[5][5] = watermark_flat[ind]
                orig_image[x:x+8, y:y+8] = subdct
                ind += 1 


    return orig_image
      

def apply_dct(image_array):
    size = image_array[0].__len__()
    all_subdct = np.empty((size, size))
    for i in range (0, size, 8):
        for j in range (0, size, 8):
            subpixels = image_array[i:i+8, j:j+8]
            subdct = dct(dct(subpixels.T, norm="ortho").T, norm="ortho")
            all_subdct[i:i+8, j:j+8] = subdct

    return all_subdct


def inverse_dct(all_subdct):
    size = all_subdct[0].__len__()
    all_subidct = np.empty((size, size))
    for i in range (0, size, 8):
        for j in range (0, size, 8):
            subidct = idct(idct(all_subdct[i:i+8, j:j+8].T, norm="ortho").T, norm="ortho")
            all_subidct[i:i+8, j:j+8] = subidct

    return all_subidct


def get_watermark(dct_watermarked_coeff, watermark_size):
    
    subwatermarks = []

    for x in range (0, dct_watermarked_coeff.__len__(), 8):
        for y in range (0, dct_watermarked_coeff.__len__(), 8):
            coeff_slice = dct_watermarked_coeff[x:x+8, y:y+8]
            subwatermarks.append(coeff_slice[5][5])

    watermark = np.array(subwatermarks).reshape(watermark_size, watermark_size)

    return watermark


def recover_watermark(image_array, model='haar', level = 1):


    coeffs_watermarked_image = process_coefficients(image_array, model, level=level)
    dct_watermarked_coeff = apply_dct(coeffs_watermarked_image[0])
    watermark_array = get_watermark(dct_watermarked_coeff, 128)
    watermark_array =  np.uint8(watermark_array)
    img = Image.fromarray(watermark_array)
    img.save(os.path.join(app.config['UPLOAD_FOLDER'], "dct-dwt_extract.png"))
    


def print_image_from_array(image_array,name):
  
    image_array_copy = image_array.clip(0, 255)
    image_array_copy = image_array_copy.astype("uint8")
    img = Image.fromarray(image_array_copy)
    img.save(os.path.join(app.config['UPLOAD_FOLDER'], name))
    return img

#Metric UJI
def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
def compare_images(img1, img2):
    # Load the images
    image1 = cv2.imread(img1, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(img2, cv2.IMREAD_GRAYSCALE)

    # Calculate Mean Squared Error (MSE)
    mse = mean_squared_error(image1, image2)

    # Calculate Structural Similarity Index (SSIM)
    ssim = structural_similarity(image1, image2)

    return mse, ssim
@app.route('/')
def home():
    # Menggunakan render_template untuk menghasilkan halaman HTML
    return render_template('index.html')
@app.route('/upload', methods=['POST'])
def upload():
    # Memproses gambar yang diunggah
    
    if 'cover' not in request.files or 'watermark' not in request.files:
        return jsonify({'error': 'Silakan unggah gambar cover dan watermark'})

    cover_image = request.files['cover']
    watermark_image = request.files['watermark']

    if cover_image.filename == '' or watermark_image.filename == '':
        return jsonify({'error': 'Gambar tidak valid'})

    # Memperbaiki cara mendapatkan nama file
    cover_filename = secure_filename(cover_image.filename)
    watermark_filename = secure_filename(watermark_image.filename)

    # Simpan file di folder yang sesuai
    cover_image.save(os.path.join(app.config['UPLOAD_FOLDER'], cover_filename))
    watermark_image.save(os.path.join(app.config['UPLOAD_FOLDER'], watermark_filename))

    # Memproses gambar yang diunggah
    cover_array = convert_image(os.path.join(app.config['UPLOAD_FOLDER'], cover_filename), 2048)
    watermark_array = convert_image(os.path.join(app.config['UPLOAD_FOLDER'], watermark_filename), 128)

    # Embed watermark pada gambar cover
    coeffs_cover = process_coefficients(cover_array, 'haar', level=1)
    dct_cover_coeff = apply_dct(coeffs_cover[0])
    dct_cover_coeff = embed_watermark(watermark_array, dct_cover_coeff)
    coeffs_cover[0] = inverse_dct(dct_cover_coeff)
    
    #reconstruction
    # watermarked_image = pywt.waverec2(coeffs_cover)
    image_array_H=pywt.waverec2(coeffs_cover, 'haar')
    watermarked_image = print_image_from_array(image_array_H,"dct-dwt_watermarked.png")
    # watermark_image.save(os.path.join(app.config['UPLOAD_FOLDER'], "dct-dwt_watermarked.png"))
    #recover_watermark 
    
    extracted_watermark_image=recover_watermark(image_array = image_array_H, model='haar', level = 1)
    
    url_embed = url_for('static', filename='dct-dwt_watermarked.png')
    url_extract = url_for('static', filename='dct-dwt_extract.png')
    
    return render_template('result.html',extracted_image_url=url_extract, embedded_image_url = url_embed )
  

# Define the main entry point of the application
if __name__ == '__main__':
    # Run the application on a local development server
    app.run(debug=True, port = 8000)

