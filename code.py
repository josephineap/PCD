import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skimage.metrics import structural_similarity as ssim
import os
import glob

# ==========================================
# 1. KONFIGURASI PARAMETER
# ==========================================
DATASET_FOLDER = 'dataset'
BLOCK_SIZE = 8
ALPHA = 30
JPEG_QUALITIES = [90, 70, 50, 30, 10]

# Mapping Frekuensi Zig-Zag
FREQ_MAP = {
    'Low':    [(0, 1), (1, 0)],
    'Mid':    [(4, 3), (3, 4)],
    'High':   [(7, 7), (7, 6)]
}

# ==========================================
# 2. FUNGSI UTILITAS
# ==========================================
def calculate_psnr(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0: return 100
    return 20 * np.log10(255.0 / np.sqrt(mse))

def calculate_nc(w_original, w_extracted):
    w1 = w_original.flatten()
    w2 = w_extracted.flatten()
    norm = np.sqrt(np.sum(w1**2)) * np.sqrt(np.sum(w2**2))
    if norm == 0: return 0
    return np.sum(w1 * w2) / norm

def dct_2d(block):
    return cv2.dct(block.astype(np.float32))

def idct_2d(block):
    return cv2.idct(block)

# ==========================================
# 3. CORE PROCESSING (Loop per Gambar)
# ==========================================
def generate_random_watermark(total_blocks):
    return np.random.choice([-1, 1], size=total_blocks)

def process_single_image(image_path, filename):
    img = cv2.imread(image_path)
    if img is None: return []

    #Padding
    h, w, _ = img.shape
    h_pad = (BLOCK_SIZE - h % BLOCK_SIZE) % BLOCK_SIZE
    w_pad = (BLOCK_SIZE - w % BLOCK_SIZE) % BLOCK_SIZE
    img_padded = cv2.copyMakeBorder(img, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
    
    # Convert YCbCr
    img_ycc = cv2.cvtColor(img_padded, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(img_ycc)
    y_float = y.astype(np.float32)
    
    h_new, w_new = y.shape
    num_blocks = (h_new // BLOCK_SIZE) * (w_new // BLOCK_SIZE)
    watermark = generate_random_watermark(num_blocks)
    
    image_results = []

    # Loop 3 Skenario Frekuensi
    for freq_name, coords in FREQ_MAP.items():

        # ==EMBEDDING==
        y_embed = y_float.copy()
        wm_idx = 0
        
        #DCT & Embed
        for i in range(0, h_new, BLOCK_SIZE):
            for j in range(0, w_new, BLOCK_SIZE):
                if wm_idx >= len(watermark): break
                block = y_embed[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
                dct = dct_2d(block)
                
                w_bit = watermark[wm_idx]
                for (cy, cx) in coords:
                    dct[cy, cx] += (ALPHA * w_bit) # Additive Rule
                
                y_embed[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = idct_2d(dct)
                wm_idx += 1
        
        # Reconstruct Image
        y_stego = np.clip(y_embed, 0, 255).astype(np.uint8)
        stego_bgr = cv2.cvtColor(cv2.merge([y_stego, cr, cb]), cv2.COLOR_YCrCb2BGR)
        stego_final = stego_bgr[:h, :w]

        # Hitung Kualitas Visual Awal
        psnr = calculate_psnr(img, stego_final)
        ssim_val = ssim(img, stego_final, channel_axis=2)
        

        # ==ATTACK JPEG & EXTRACT==
        for q in JPEG_QUALITIES:
            # Save & Load JPEG (Simulasi Serangan)
            temp_name = f"temp_{filename}_{freq_name}_{q}.jpg"
            cv2.imwrite(temp_name, stego_final, [cv2.IMWRITE_JPEG_QUALITY, q])
            attacked_img = cv2.imread(temp_name)
            
            # Extraction Process
            # (Preprocessing ulang attacked image)
            att_padded = cv2.copyMakeBorder(attacked_img, 0, h_pad, 0, w_pad, cv2.BORDER_REFLECT)
            y_att = cv2.split(cv2.cvtColor(att_padded, cv2.COLOR_BGR2YCrCb))[0].astype(np.float32)
            
            # Original Y untuk referensi ekstraksi
            y_orig_ref = y_float
            
            extracted_wm = []
            wm_idx_ex = 0
            
            for i in range(0, h_new, BLOCK_SIZE):
                for j in range(0, w_new, BLOCK_SIZE):
                    if wm_idx_ex >= len(watermark): break
                    dct_att = dct_2d(y_att[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE])
                    dct_orig = dct_2d(y_orig_ref[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE])
                    
                    diff_sum = 0
                    for (cy, cx) in coords:
                        diff_sum += (dct_att[cy, cx] - dct_orig[cy, cx])
                    
                    extracted_wm.append(1 if diff_sum > 0 else -1)
                    wm_idx_ex += 1
            
            # Hitung NC
            nc = calculate_nc(watermark, np.array(extracted_wm))
            
            image_results.append({
                'Image': filename,
                'Frequency': freq_name,
                'JPEG_Quality': q,
                'PSNR': psnr,
                'SSIM': ssim_val,
                'NC': nc
            })
            
            if os.path.exists(temp_name): os.remove(temp_name)

    return image_results

# ==========================================
# 4. MAIN PROGRAM (BATCH PROCESSING)
# ==========================================
if __name__ == "__main__":
    #Cek Folder
    if not os.path.exists(DATASET_FOLDER):
        os.makedirs(DATASET_FOLDER)
        print(f"Folder '{DATASET_FOLDER}' baru dibuat. Masukkan gambar dulu ke situ!")
    else:
        #Mengambil semua gambar (jpg, png, tif, bmp)
        extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.bmp', '*.tiff']
        image_files = []
        for ext in extensions:
            image_files.extend(glob.glob(os.path.join(DATASET_FOLDER, ext)))
            
        if not image_files:
            print(f"Tidak ada gambar di folder '{DATASET_FOLDER}'.")
        else:
            print(f"Ditemukan {len(image_files)} gambar. Mulai Batch Processing...")
            
            all_data = []
            
            for img_path in image_files:
                fname = os.path.basename(img_path)
                print(f"Processing: {fname}...")
                res = process_single_image(img_path, fname)
                all_data.extend(res)
                
            # ==========================
            # 5. AGGREGATE & PLOT
            # ==========================
            df = pd.DataFrame(all_data)
            
            # Simpan Data Mentah
            df.to_csv('hasil_lengkap_per_gambar.csv', index=False)
            
            df_avg = df.groupby(['Frequency', 'JPEG_Quality']).mean(numeric_only=True).reset_index()
            df_avg.to_csv('hasil_rata_rata.csv', index=False)
            
            print("\nAnalisis Selesai!")
            print("1. Data detail tersimpan di 'hasil_lengkap_per_gambar.csv'")
            print("2. Data rata-rata tersimpan di 'hasil_rata_rata.csv'")
            
            #Plot Grafik Rata-Rata
            plt.figure(figsize=(10, 6))
            markers = {'Low': 'o', 'Mid': 's', 'High': '^'}
            colors = {'Low': 'blue', 'Mid': 'green', 'High': 'red'}
            
            for freq in ['Low', 'Mid', 'High']:
                subset = df_avg[df_avg['Frequency'] == freq]
                subset = subset.sort_values('JPEG_Quality', ascending=False)
                
                plt.plot(subset['JPEG_Quality'], subset['NC'],
                         marker=markers[freq], color=colors[freq],
                         linewidth=2, label=f'{freq} Freq (Avg)')
            
            plt.title(f'Average Robustness Analysis ({len(image_files)} Images)')
            plt.xlabel('JPEG Quality Factor (Q)')
            plt.ylabel('Average Normalized Correlation (NC)')
            plt.ylim(0, 1.1)
            plt.grid(True, linestyle='--')
            plt.legend()
            plt.gca().invert_xaxis()
            
            plt.savefig('grafik_rata_rata.png', dpi=300)
            print("3. Grafik rata-rata tersimpan di 'grafik_rata_rata.png'")
            plt.show()