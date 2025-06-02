# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech
## Business Understanding
Jaya Jaya Institut merupakan salah satu institusi pendidikan perguruan yang telah berdiri sejak tahun 2000. Hingga saat ini ia telah mencetak banyak lulusan dengan reputasi yang sangat baik. Akan tetapi, terdapat banyak juga siswa yang tidak menyelesaikan pendidikannya alias dropout.

Jumlah dropout yang tinggi ini tentunya menjadi salah satu masalah yang besar untuk sebuah institusi pendidikan. Oleh karena itu, Jaya Jaya Institut ingin mendeteksi secepat mungkin siswa yang mungkin akan melakukan dropout sehingga dapat diberi bimbingan khusus. 

## Permasalahan Bisnis
1.   Bagaimanakah proporsi dropout dan tidak droput dari kelompok mahasiswa berdasarkan sebera lancar dalam membayar uang kuliah?
2.   Bagaimanakah hubungan performa di semester satu (number of courses approved) mempengaruhi apakah mahasiswa akan dropout atau tidak?
3.   Bagaimanakah kelompok usia mempengaruhi kemungkinan drop out?
4.  Bagaimanakah dampak dari penerimaan beasiswa terhadap kemungkinan drop out?

### Cakupan Proyek
* menjawab pertanyaan bisnis menggunakan analisis data sederhana
* membuat business dashboard
* membuat model machine learning untuk memprediksi apakah karyawan akan keluar perusahaan
* mendploy solusi machine learning ke streamlit 

### Persiapan
#### Sumber data: https://github.com/dicodingacademy/dicoding_dataset/blob/main/students_performance/README.md

#### Setup environment:
```
# buat venv
python -m venv bpds-final-env

# aktifkan venv
.\bpds-final-env\Scripts\activate

# install dependencies
pip install -r requirements.txt

# jalankan streamlit secara local
streamlit run app.py

# link streamlit


# metabase credentials
username : medianosandie@gmail.com
password : passMetabase1?
```


## Business Dashboard
Pada dashboard terdapat 4 visualisasi yang dapat digunakan untuk memonitor beberapa metrics
Berikut keempat visualisasi yang dimaksud:
1.proporsi dropout dan tidak droput dari kelompok mahasiswa berdasarkan sebera lancar dalam membayar uang kuliah
2.perbandingan performa di semester satu (number of courses approved) pada kelompok dropout dan not-dropout
3.perbandingan dropout rate dari berbagai kelompok usia
4.perbandingan dropout rate dari kelompok pemegang beasiswa vs non-pemegang-beasiswa

## Conclusion
1.  Keterlambatan Pembayaran Uang Kuliah: Mahasiswa yang menunggak memiliki risiko dropout sangat tinggi (87%) dibandingkan yang lancar membayar (25%).
2.  Performa Akademik Semester Pertama: Mahasiswa yang dropout rata-rata hanya menyelesaikan 2,6 mata kuliah, jauh di bawah mahasiswa non-dropout (5,7). Performa awal sangat krusial.
3.  Kelompok Usia: Mahasiswa usia 25 tahun ke atas memiliki rasio dropout dua kali lebih tinggi, bahkan melebihi 50%, dibandingkan usia muda.
4.  Status Beasiswa: Penerima beasiswa menunjukkan tingkat dropout yang jauh lebih rendah (12%) dibandingkan non-penerima (39%).
Berdasarkan analisis data, ditemukan bahwa faktor-faktor seperti
*   keterlambatan pembayaran uang kuliah, performa rendah di semester awal,
*   usia yang lebih tua saat masuk,
*   dan ketidakadaan beasiswa
Oleh karena itu, pendekatan pencegahan dropout perlu ditargetkan secara spesifik kepada kelompok rentan ini dengan dukungan finansial, akademik, dan fleksibilitas belajar.

### Rekomendasi Action Items 
1.  Finansial:
    *   Buat sistem peringatan dini untuk penunggakan.
    *   Sediakan skema cicilan & konsultasi keuangan.

2.  Akademik:
    *   Identifikasi mahasiswa dengan performa rendah.
    *   Tawarkan program mentoring dan remedial.

3.  Kelompok Usia 25+:
    *   Fasilitasi kelas fleksibel & dukungan karir.
    *   Adakan pelatihan manajemen waktu dan studi mandiri.

4.  Beasiswa:
    *   Perluas program beasiswa bagi mahasiswa rentan.
    *   Berikan insentif berbasis performa untuk non-penerima.

