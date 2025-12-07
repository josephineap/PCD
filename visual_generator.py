import cv2
import numpy as np
import os
import glob

# ==========================================
# KONFIGURASI (SAMA DENGAN EKSPERIMEN UTAMA)
# ==========================================
DATASET_FOLDER = 'dataset'
BLOCK_SIZE = 8
ALPHA = 30
FREQ_MAP = {
    'Low':    [(0, 1), (1, 0)],
    'Mid':    [(4, 3), (3, 4)],
    'High':   [(7, 7), (7, 6)]
}

#Koordinat Zoom/Crop
#CROP_Y, CROP_X = 100, 150  (Baboon area Mata/Hidung, karena ada strukturnya)
CROP_Y, CROP_X = 100, 150
CROP_SIZE = 100

def dct_2d(block): return cv2.dct(block.astype(np.float32))
def idct_2d(block): return cv2.idct(block)

def generate_visuals():
    # 1.Ambil 1 Gambar (Otomatis ambil file pertama di folder --> yaitu Baboon)
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(DATASET_FOLDER, ext)))
    
    if not files:
        print("Folder dataset kosong!")
        return

    #Ambil gambar pertama (misal Baboon)
    target_img_path = files[0]
    print(f"Menggunakan gambar sampel: {target_img_path}")
    
    img = cv2.imread(target_img_path)
    
    if img is None:
        print("Gagal membaca gambar.")
        return
        
    h, w, _ = img.shape
    
    #--- SIMPAN GAMBAR ASLI (FULL & ZOOM) ---
    print("Generating visual untuk: Original...")
    cv2.imwrite("Paper_Fig_Original.png", img)
    
    # Zoom Asli (Crop & Upscale)
    crop_orig = img[CROP_Y:CROP_Y+CROP_SIZE, CROP_X:CROP_X+CROP_SIZE]
    crop_orig_upscale = cv2.resize(crop_orig, (300, 300), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite("Paper_Fig_Zoom_Original.png", crop_orig_upscale)
    
    #--- PROSES WATERMARKING ---
    #Padding
    h_pad = (BLOCK_SIZE - h % BLOCK_SIZE) % BLOCK_SIZE
    w_pad = (BLOCK_SIZE - w % BLOCK_SIZE) % BLOCK_SIZE
    img_padded = cv2.copyMakeBorder(img, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    
    #Proses ke YCbCr
    img_ycc = cv2.cvtColor(img_padded, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_ycc)
    y_float = y.astype(np.float32)
    
    #Buat Watermark Acak
    h_new, w_new = y.shape
    num_blocks = (h_new // BLOCK_SIZE) * (w_new // BLOCK_SIZE)
    watermark = np.random.choice([-1, 1], size=num_blocks)

    # 2.Loop untuk Low, Mid, High
    for freq_name, coords in FREQ_MAP.items():
        print(f"Generating visual untuk: {freq_name}...")
        y_embed = y_float.copy()
        wm_idx = 0
        
        #Embedding Process
        for i in range(0, h_new, BLOCK_SIZE):
            for j in range(0, w_new, BLOCK_SIZE):
                if wm_idx >= len(watermark): break
                block = y_embed[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
                dct = dct_2d(block)
                w_bit = watermark[wm_idx]
                for (cy, cx) in coords:
                    dct[cy, cx] += (ALPHA * w_bit)
                y_embed[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = idct_2d(dct)
                wm_idx += 1
        
        #Reconstruct
        y_stego = np.clip(y_embed, 0, 255).astype(np.uint8)
        stego_bgr = cv2.cvtColor(cv2.merge([y_stego, cr, cb]), cv2.COLOR_YCrCb2BGR)
        stego_final = stego_bgr[:h, :w]
        
        #Simpan Full Image
        filename_full = f"Paper_Fig_Stego_{freq_name}.png"
        cv2.imwrite(filename_full, stego_final)
        
        #Simpan Zoom/Cropped Image
        crop_img = stego_final[CROP_Y:CROP_Y+CROP_SIZE, CROP_X:CROP_X+CROP_SIZE]
        crop_upscale = cv2.resize(crop_img, (300, 300), interpolation=cv2.INTER_NEAREST)
        
        filename_crop = f"Paper_Fig_Zoom_{freq_name}.png"
        cv2.imwrite(filename_crop, crop_upscale)
        
    print("\nSELESAI! Cek folder Anda.")
    print("Sekarang sudah ada file 'Paper_Fig_Zoom_Original.png' juga.")

if __name__ == "__main__":
    generate_visuals()