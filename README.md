# Distributed Image Processing

Program pemrosesan gambar Master–Worker menggunakan Python, TCP, UDP discovery, NumPy, OpenCV, dan Pillow. Master membagi gambar menjadi tile 2D, menjadwalkan task ke Worker, lalu menyusun hasilnya.

## 1. Siapkan source code

Salin folder proyek ke komputer Master dan setiap komputer Worker. Semua komputer harus memakai versi source yang sama. Di Worker, `master.py` dan folder `before/` tidak dibutuhkan, tetapi menyalin seluruh folder adalah cara termudah.

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

Worker membatasi penerimaan task aktif sesuai jumlah slot prosesnya. Jika semua slot sedang digunakan—misalnya oleh Master lain—Worker menjawab `busy` dan Master langsung menolak/membatalkan job tanpa menunggu atau mengalihkan task diam-diam.

Setelah Worker aktif, pada komputer Master jalankan `python master.py`. Master meminta gambar dan operasi, lalu meminta faktor skala jika memilih Upscale atau ukuran search window jika memilih NLM. Setelah itu Master meminta mode pemrosesan, pilihan Worker, dan scheduler. Untuk gambar di `before/`, cukup ketik nama file seperti `anime.png`; path relatif `before/anime.png` dan path absolut juga diterima.

Pilihan operasi gambar:

- **Upscale:** memperbesar dan mempertajam gambar seperti pipeline sebelumnya; Master meminta faktor skala.
- **NLM:** denoising warna dengan ukuran output sama seperti input; tidak perlu membuat gambar noise terlebih dahulu. Search window default 21 dan harus bilangan ganjil minimal 7. Nilai lebih besar menambah kerja komputasi dan memperlebar konteks tile. Master menyimpan hasil dengan akhiran `_nlm_sw<N>.png`.

Contoh tanpa menu interaktif: `python master.py --input before/anime.png --operation nlm --nlm-search-window 21 --mode sequential`. Untuk Upscale, tambahkan `--operation upscale --scale 2.0`.

Mode pemrosesan:

- **Sequential:** satu proses mengolah semua tile satu per satu sebagai baseline.
- **Local parallel:** Master membagi gambar ke beberapa proses lokal.
- **Distributed:** Worker terpilih mengolah task secara bersamaan; Master menggabungkan hasil.

Scheduler:

- **Dynamic:** task berikutnya diberikan ke slot Worker yang baru selesai, jadi pembagian mengikuti siapa yang selesai lebih cepat.
- **Static:** semua task ditentukan lebih dahulu untuk slot proses. Jumlahnya tidak harus sama per perangkat: Worker dengan slot proses lebih banyak mendapat bagian task lebih besar. Static berarti penugasannya sudah ditentukan di awal, bukan pembagian merata per Worker.

Tabel Worker menampilkan jumlah task yang selesai dan kolom `FAILED` untuk task yang gagal pada Worker tempat task itu pertama kali ditugaskan, termasuk task yang kemudian berhasil dialihkan. Tabel Task menunjukkan status setiap task: `SUCCESS`, `REROUTED TO <Worker>`, atau `FAILED`. Jika task dialihkan, kolom Worker menunjukkan Worker asal dan status menyebut Worker tujuan.

Setelah memilih scheduler, Master menanyakan apakah fault tolerance diaktifkan. Jika **tidak**, task yang gagal tidak di-retry; task lain tetap diproses dan gambar parsial disimpan dengan area task gagal berwarna magenta. Jika **ya**, Master mencoba Worker lain lebih dahulu, menunggu dan mencoba lagi jika slotnya sedang sibuk, lalu mencoba Worker asal sebagai pilihan terakhir. Setiap putaran retry diberi jeda yang meningkat dan dibatasi empat putaran agar job tidak menunggu selamanya jika semua Worker mati. Log menampilkan task dan Worker yang gagal, target retry, jeda, serta hasil pemulihannya. Jika seluruh percobaan gagal, area task itu tetap ditandai magenta. Untuk menggunakan NLM, salin source terbaru `master.py`, `worker.py`, dan `image_pipeline.py` ke semua komputer lalu restart seluruh Worker. Worker lama yang belum mengiklankan dukungan NLM akan ditolak oleh Master untuk operasi tersebut.

Semua mode memakai pembagian tile 2D yang sama: inti default 128×128 piksel sumber. Upscale memakai konteks default 16 piksel; NLM menambah konteks sesuai search window dan ukuran patch filter. Sequential memproses tile satu per satu di satu proses; local-parallel dan distributed memproses task tile tersebut secara paralel. Master hanya menyimpan inti tile saat menyatukan hasil. Jumlah task berasal dari ukuran gambar: `ceil(lebar/ukuran_tile) × ceil(tinggi/ukuran_tile)`, bukan dari jumlah Worker. Contoh gambar 2048×2048 menghasilkan grid 16×16 atau 256 task; gambar 2048×1024 menghasilkan grid 16×8 atau 128 task. Master mencetak ukuran inti, grid, jumlah task, konteks, dan slot proses yang terdeteksi sebelum dispatch. Slot menentukan berapa task yang berjalan bersamaan; task lain menunggu giliran. Untuk ukuran inti manual, gunakan `--tile-size N`. Dengan gambar, operasi, parameter, dan ukuran tile yang sama, pengujian mode memakai task yang sama. Hasil tile diproses sendiri-sendiri; periksa sambungan gambar karena filter lokal dapat menghasilkan perbedaan di batas tile.

Output gambar disimpan di `after/`. Untuk mode local-parallel dan distributed, Master juga menjalankan sequential baseline pada input dan operasi yang sama, dengan tile yang sama, lalu menampilkan metrik utama dalam tabel. Waktu jaringan menggabungkan waktu Worker menerima task dan mengirim hasil. Di semua tabel, Network mengurangi waktu yang bertepatan dengan komputasi di Worker mana pun, memakai interval komputasi job yang sama dengan tabel T_RUN. Dengan begitu satu task atau Worker tidak bisa melaporkan Network lebih lama daripada Network global hanya karena komputasi Worker lain sedang berlangsung. Task-task dan Worker-Worker yang melakukan transfer bersamaan masih memiliki interval Network yang saling tumpang tindih; jangan jumlahkan kolom Network per baris untuk memperoleh Network global. Label `Other (whole job)` adalah sisa wall time seluruh job, termasuk persiapan dan stitching. `OTHER (WORKER)` adalah sisa dari waktu aktif satu Worker, sedangkan `OTHER (TASK)` adalah sisa dari durasi satu task, termasuk waktu tunggu dan overhead task. Karena cakupannya berbeda, nilai Other antar tabel tidak harus sama dan tidak untuk dijumlahkan. Waktu jaringan Worker diukur memakai monotonic timer pada Worker untuk menerima task, dan timer Master untuk menerima hasil. Tabel task menampilkan ukuran data task dan hasil, waktu jaringan eksklusif, serta laju efektif MB/s. Laporan teks otomatis dibuat di `logs/` dengan nama berisi hostname Master, resolusi, operasi/parameter, mode, scheduler, jumlah Worker, dan waktu run.

## 5. Uji fault tolerance

Fault tolerance berbeda dari load balancing. Untuk mengujinya, pilih **Yes** pada prompt fault tolerance, mulai distributed job dengan cukup banyak task, lalu hentikan proses `worker.py` pada salah satu komputer saat job berlangsung. Master mendeteksi task yang gagal/timeout dan mengirim ulang task ke Worker lain. Terminal menunjukkan task dan Worker asal yang gagal serta tujuan pengalihan. Jika retry berhasil, gambar lengkap tersimpan; jika tidak, area yang gagal ditandai magenta. Catat log Master dan Worker sebagai bukti.

## 6. Skenario evaluasi UAS

Gunakan input gambar persegi yang sama untuk membandingkan sequential, local parallel, static, dan dynamic. Ulangi distributed processing dengan 1, 2, 4, 6, dan 9 Worker. Simpan bukti terminal Master dan Worker. Persyaratan demo 1 Master + 9 Worker harus dibuktikan dengan komputer fisik; simulasi lokal tidak menggantikannya.

Jika Worker tidak terdeteksi, pastikan process `worker.py` aktif, firewall mengizinkan TCP 5000 dan UDP 5001, serta perangkat berada pada LAN/broadcast domain yang sama. Jika broadcast jaringan diblokir, gunakan opsi alamat Worker manual pada command-line.
