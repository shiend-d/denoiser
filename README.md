# Distributed Image Processing

Program pemrosesan gambar Master–Worker menggunakan Python, TCP, UDP discovery, NumPy, OpenCV, dan Pillow. Master membagi gambar menjadi tile 2D, menjadwalkan task ke Worker, lalu menyusun hasilnya.

## 1. Siapkan source code

Salin folder proyek ke komputer Master dan setiap komputer Worker. Semua komputer harus memakai versi source yang sama. Worker tidak membutuhkan gambar di folder `input/`, tetapi menyalin seluruh folder adalah cara termudah.

## 2. Instal dependensi

Pasang dependensi di Master dan setiap Worker. Jalankan perintah instalasi dan aplikasi dengan interpreter Python yang sama. `requirements.txt` mencakup NumPy, OpenCV, Pillow, dan psutil.

### NixOS

```bash
nix develop
python -c "import numpy, cv2, PIL, psutil; print('Dependencies ready')"
```

`nix develop` sudah menyediakan Python, pustaka native, dan dependensi proyek. Jalankan `python worker.py` pada Worker atau `python master.py` pada Master dari shell tersebut. Jangan aktifkan `.venv` atau install wheel lewat pip di luar dev shell. Jika sebelumnya membuat `.venv`, keluar dari environment itu (`deactivate`), hapus folder `.venv`, buka shell baru, lalu jalankan `nix develop`.

### Linux lain

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -3.14 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Jika PowerShell menolak aktivasi environment, jalankan `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`, lalu aktifkan kembali `.venv`. Jika launcher `py -3.14` tidak tersedia, buat environment dengan versi Python yang terpasang menggunakan `python -m venv .venv`.

Verifikasi dependensi pada platform mana pun:

```bash
python -c "import numpy, cv2, PIL, psutil; print('Dependencies ready')"
```

## 3. Siapkan jaringan

Master dan Worker harus terhubung ke LAN yang sama dan mengizinkan broadcast UDP untuk discovery. Izinkan koneksi masuk **TCP 5000** (task/result) dan **UDP 5001** (discovery) pada firewall Worker.

Di NixOS, tambahkan port berikut ke konfigurasi firewall, lalu rebuild sistem:

```nix
networking.firewall.allowedTCPPorts = [ 5000 ];
networking.firewall.allowedUDPPorts = [ 5001 ];
```

Di Windows, jalankan PowerShell sebagai Administrator:

```powershell
New-NetFirewallRule -DisplayName "Upscayl Worker TCP 5000" -Direction Inbound -Protocol TCP -LocalPort 5000 -Action Allow -Profile Private
New-NetFirewallRule -DisplayName "Upscayl Discovery UDP 5001" -Direction Inbound -Protocol UDP -LocalPort 5001 -Action Allow -Profile Private
```

## 4. Jalankan Worker lalu Master

Di setiap komputer Worker, masuk ke folder proyek, aktifkan `.venv` jika dibuat, lalu jalankan:

```bash
python worker.py
```

Worker memakai seluruh core CPU yang terdeteksi secara default. Biarkan terminal dan proses Worker tetap aktif.

Worker membatasi penerimaan task aktif sesuai jumlah slot prosesnya. Jika semua slot sedang digunakan—misalnya oleh Master lain—Worker menjawab `busy`. Dengan fault tolerance aktif, Master menunggu lalu mencoba task lagi; jika nonaktif, task ditandai gagal dan wilayahnya menjadi magenta pada hasil parsial.

Letakkan gambar asli di folder `input/`. Jalankan `python master.py` pada Master. Masukkan nama file seperti `anime.png` atau path gambar. Master membuat salinan noisy deterministik bernama `input/anime-before.png`, lalu semua mode memproses salinan yang sama. Gambar asli tidak diubah. Noise Gaussian ditetapkan pada sigma 20 dengan seed 42 agar pengujian konsisten. Masukkan search window NLM berupa angka ganjil minimal 7; nilai default 31 bisa dipakai dengan menekan Enter. Hasil NLM disimpan sebagai `output/anime-after-nlm-sw31.png`.

Contoh tanpa menu interaktif: `python master.py --input input/anime.png --mode sequential --nlm-search-window 31`. Kekuatan filter `h` dan `hColor` ditetapkan 10 agar perbandingan antar mode konsisten.

Mode pemrosesan:

- **Sequential:** satu proses mengolah semua tile satu per satu sebagai baseline.
- **Local parallel:** Master membagi gambar ke beberapa proses lokal.
- **Distributed:** Worker terpilih mengolah task secara bersamaan; Master menggabungkan hasil.

Scheduler:

- **Dynamic:** task berikutnya diberikan ke slot Worker yang baru selesai, jadi pembagian mengikuti siapa yang selesai lebih cepat.
- **Static:** semua task ditentukan lebih dahulu untuk slot proses. Jumlahnya tidak harus sama per perangkat: Worker dengan slot proses lebih banyak mendapat bagian task lebih besar. Static berarti penugasannya sudah ditentukan di awal, bukan pembagian merata per Worker.

Tabel Worker menampilkan jumlah task yang selesai dan kolom `FAILED` untuk task yang gagal pada Worker tempat task itu pertama kali ditugaskan, termasuk task yang kemudian berhasil dialihkan. Tabel Task menunjukkan status setiap task: `SUCCESS`, `REROUTED TO <Worker>`, atau `FAILED`. Jika task dialihkan, kolom Worker menunjukkan Worker asal dan status menyebut Worker tujuan.

Setelah memilih scheduler, Master menanyakan apakah fault tolerance diaktifkan. Jika **tidak**, task yang gagal tidak di-retry; task lain tetap diproses dan gambar parsial disimpan dengan area task gagal berwarna magenta. Jika **ya**, Master mencoba Worker lain lebih dahulu, menunggu dan mencoba lagi jika slotnya sedang sibuk, lalu mencoba Worker asal sebagai pilihan terakhir. Setiap putaran retry diberi jeda yang meningkat dan dibatasi empat putaran agar job tidak menunggu selamanya jika semua Worker mati. Log menampilkan task dan Worker yang gagal, target retry, jeda, serta hasil pemulihannya. Jika seluruh percobaan gagal, area task itu tetap ditandai magenta. Salin source terbaru `master.py`, `worker.py`, `image_pipeline.py`, dan `config.py` ke semua komputer lalu restart seluruh Worker. Worker lama atau Worker dengan format task yang berbeda akan ditolak oleh Master.

Semua mode memakai pembagian tile 2D yang sama: inti default 128×128 piksel dan konteks NLM minimal 16 piksel, diperbesar bila search window lebih besar. Sequential memproses tile satu per satu di satu proses; local-parallel dan distributed memproses task yang sama secara paralel. Master hanya menyimpan inti tile saat menyatukan hasil. Jumlah task berasal dari ukuran gambar: `ceil(lebar/ukuran_tile) × ceil(tinggi/ukuran_tile)`, bukan dari jumlah Worker. Contoh gambar 2048×2048 menghasilkan grid 16×16 atau 256 task; gambar 2048×1024 menghasilkan grid 16×8 atau 128 task. Slot menentukan berapa task yang berjalan bersamaan; task lain menunggu giliran. Untuk ukuran inti manual, gunakan `--tile-size N`. Noise dibuat sekali di Master sebelum baseline/run dan file `-before` yang sama digunakan untuk semua mode. Waktu pembuatan noise ditampilkan di Overall Analysis, terpisah dari T_run.

Hasil disimpan di `output/<nama>-after-nlm-sw<ukuran>.png`; input noisy tersimpan di `input/<nama>-before.png`. Sequential, local-parallel, dan distributed menjalankan filter serta task yang sama. Noise preparation dilaporkan terpisah dan tidak masuk ke T_run; T_run mengukur task preparation, transfer/komputasi bila distributed, stitching, dan overhead proses. Waktu jaringan menggabungkan waktu Worker menerima task dan mengirim hasil. Di semua tabel, Network mengurangi waktu yang bertepatan dengan komputasi di Worker mana pun, memakai interval komputasi job yang sama dengan tabel T_RUN. Task-task dan Worker-Worker yang melakukan transfer bersamaan masih memiliki interval Network yang saling tumpang tindih; jangan jumlahkan kolom Network per baris untuk memperoleh Network global. `Other (whole job)` adalah sisa wall time seluruh job. Tabel Worker menampilkan jumlah task, kegagalan, Network, Compute, CPU, RAM, dan retry. Tabel Task menampilkan Network, Compute, dan total durasi task; baris task dapat tumpang tindih. Overall Analysis dan log mencatat mode, scheduler, status fault tolerance, search window, serta metrik waktu. Nama laporan di `logs/` memuat hostname Master, resolusi, search window, mode, scheduler, jumlah Worker, dan waktu run.

## 5. Uji fault tolerance

Fault tolerance berbeda dari load balancing. Untuk mengujinya, pilih **Yes** pada prompt fault tolerance, mulai distributed job dengan cukup banyak task, lalu hentikan proses `worker.py` pada salah satu komputer saat job berlangsung. Master mendeteksi task yang gagal/timeout dan mengirim ulang task ke Worker lain. Terminal menunjukkan task dan Worker asal yang gagal serta tujuan pengalihan. Jika retry berhasil, gambar lengkap tersimpan; jika tidak, area yang gagal ditandai magenta. Catat log Master dan Worker sebagai bukti.

## 6. Skenario evaluasi UAS

Gunakan input gambar persegi yang sama untuk membandingkan sequential, local parallel, static, dan dynamic. Ulangi distributed processing dengan 1, 2, 4, 6, dan 9 Worker. Simpan bukti terminal Master dan Worker. Persyaratan demo 1 Master + 9 Worker harus dibuktikan dengan komputer fisik; simulasi lokal tidak menggantikannya.

Jika Worker tidak terdeteksi, pastikan process `worker.py` aktif, firewall mengizinkan TCP 5000 dan UDP 5001, serta perangkat berada pada LAN/broadcast domain yang sama. Jika broadcast jaringan diblokir, gunakan opsi alamat Worker manual pada command-line.
