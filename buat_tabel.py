import pandas as pd

# 1.membaca dari file yg sudah ada
df = pd.read_csv('hasil_rata_rata.csv')

# 2.Mengambil kolom penting saja & Buang duplikat
# (Karena PSNR/SSIM itu sama untuk semua Quality, jadi ambil rata-ratanya per frekuensi)
tabel_rekap = df.groupby('Frequency')[['PSNR', 'SSIM']].mean().reset_index()

# 3.Format
tabel_rekap['Frequency'] = tabel_rekap['Frequency'] + ' Frequency'

#Bulatkan angka (2 desimal untuk PSNR, 4 desimal untuk SSIM)
tabel_rekap['PSNR'] = tabel_rekap['PSNR'].round(2)
tabel_rekap['SSIM'] = tabel_rekap['SSIM'].round(4)

tabel_rekap.columns = ['Lokasi Frekuensi', 'Rata-rata PSNR (dB)', 'Rata-rata SSIM']

# 4. Tampilkan & Simpan
print("\n=== TABEL 1 UNTUK PAPER ===")
print(tabel_rekap.to_string(index=False))

#Menyimpan ke Excel
tabel_rekap.to_excel('Tabel_1_Paper.xlsx', index=False)
print("\nTabel juga sudah disimpan sebagai 'Tabel_1_Paper.xlsx'")