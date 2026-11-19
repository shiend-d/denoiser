# Distributed Image Upscaler

Program pemrosesan gambar Master–Worker menggunakan Python, TCP, UDP discovery, NumPy, OpenCV, dan Pillow. Master membagi gambar menjadi strip, menjadwalkan task ke Worker, lalu menyusun hasilnya.

## 1. Siapkan source code

Salin folder proyek ke komputer Master dan setiap komputer Worker. Semua komputer harus memakai versi source yang sama. Di Worker, `master.py` dan folder `before/` tidak dibutuhkan, tetapi menyalin seluruh folder adalah cara termudah.

## 2. Instal dependensi

Pasang dependensi di Master dan setiap Worker. Jalankan perintah instalasi dan aplikasi dengan interpreter Python yang sama. `requirements.txt` mencakup NumPy, OpenCV, Pillow, dan psutil.

### Linux (termasuk NixOS)

Untuk NixOS, `nix develop` menyediakan Python dan pustaka sistem proyek. Setelah masuk ke dev shell, instal dependensi Python:

```bash
nix develop
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Di distribusi Linux lain, lewati `nix develop` lalu jalankan empat perintah berikut dari folder proyek:

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

Setelah Worker aktif, pada komputer Master jalankan `python master.py`. Master akan memindai Worker, menampilkan daftar perangkat, lalu meminta path gambar, faktor skala, mode pemrosesan, pilihan Worker, dan scheduler. Path gambar harus menunjuk ke file yang ada di komputer Master, misalnya `before/anime.png`.

Mode pemrosesan:

- **Sequential:** satu proses mengolah gambar penuh sebagai baseline.
- **Local parallel:** Master membagi gambar ke beberapa proses lokal.
- **Distributed:** Worker terpilih mengolah task secara bersamaan; Master menggabungkan hasil.

Scheduler:

- **Dynamic:** task berikutnya diberikan ke slot Worker yang baru selesai.
- **Static:** task dibagikan terlebih dahulu ke slot Worker.

Output gambar disimpan di `after/`. Untuk mode local-parallel dan distributed, Master juga menjalankan sequential baseline pada input yang sama. Terminal menampilkan execution time, speedup, efficiency, throughput, waktu komputasi dan jumlah task per Worker, rata-rata CPU/memory Worker, jumlah retry, serta recovery time. CPU dan memory Worker diambil saat task berjalan; task yang sangat singkat mungkin menghasilkan sedikit sampel.

## 5. Arti simulasi kehilangan Worker

Pilihan **Simulate scheduler removing a Worker** hanya menguji perubahan scheduler. Contoh: jika diisi `3`, setelah tiga task selesai Master berhenti memberikan task baru ke Worker yang menyelesaikan task ketiga. Proses Worker tersebut **tidak dimatikan** dan koneksinya **tidak diputus**. Karena itu, pilihan ini bukan bukti fault detection atau recovery dari Worker yang benar-benar gagal.

Untuk uji kegagalan nyata, mulai distributed job dengan cukup banyak task, lalu hentikan proses `worker.py` pada salah satu komputer saat job berlangsung. Master semestinya mendeteksi task yang gagal/timeout dan mencoba mengirim ulang task ke Worker lain. Catat log Master dan Worker sebagai bukti.

## 6. Skenario evaluasi UAS

Gunakan input gambar persegi yang sama untuk membandingkan sequential, local parallel, static, dan dynamic. Ulangi distributed processing dengan 1, 2, 4, 6, dan 9 Worker. Simpan bukti terminal Master dan Worker. Persyaratan demo 1 Master + 9 Worker harus dibuktikan dengan komputer fisik; simulasi lokal tidak menggantikannya.

Jika Worker tidak terdeteksi, pastikan process `worker.py` aktif, firewall mengizinkan TCP 5000 dan UDP 5001, serta perangkat berada pada LAN/broadcast domain yang sama. Jika broadcast jaringan diblokir, gunakan opsi alamat Worker manual pada command-line.
