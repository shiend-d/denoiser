# Distributed NLM Image Denoiser

Panduan menyiapkan dan menjalankan aplikasi pemrosesan gambar NLM dengan satu Master dan satu atau lebih Worker.

## Persiapan source code

Salin atau clone folder proyek ke komputer Master dan semua komputer Worker. Gunakan source code versi yang sama pada seluruh komputer. Gambar input hanya perlu tersedia di komputer Master.

## Instalasi

Pasang dependensi pada komputer Master dan Worker. Gunakan Python yang sama untuk instalasi dan menjalankan program.

### NixOS

Dari folder proyek, jalankan:

```bash
nix develop
```

Shell ini menyediakan Python dan dependensi proyek. Jalankan program dari shell tersebut; tidak perlu membuat `.venv` atau memasang paket melalui pip.

### Linux

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

Jika `py -3.14` tidak tersedia, gunakan Python yang telah terpasang:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Jika PowerShell menolak aktivasi `.venv`, jalankan PowerShell sebagai pengguna biasa lalu izinkan skrip lokal:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

Verifikasi instalasi pada Linux atau Windows:

```bash
python -c "import numpy, cv2, PIL, psutil; print('Dependencies ready')"
```

## Persiapan jaringan

Hubungkan Master dan Worker ke jaringan lokal yang sama. Izinkan koneksi masuk berikut pada firewall setiap Worker:

- TCP `5000` untuk task dan hasil gambar.
- UDP `5001` untuk pencarian Worker otomatis.

Di Windows, buka PowerShell sebagai Administrator dan jalankan:

```powershell
New-NetFirewallRule -DisplayName "NLM Worker TCP 5000" -Direction Inbound -Protocol TCP -LocalPort 5000 -Action Allow -Profile Private
New-NetFirewallRule -DisplayName "NLM Discovery UDP 5001" -Direction Inbound -Protocol UDP -LocalPort 5001 -Action Allow -Profile Private
```

## Menjalankan Worker

Di setiap komputer Worker, buka folder proyek, aktifkan environment Python jika menggunakannya, lalu jalankan:

```bash
python worker.py
```

## Menjalankan Master

Simpan gambar asli di folder `input/` pada komputer Master, lalu jalankan:

```bash
python master.py
```

Masukkan nama file, misalnya `anime.png`, atau path gambarnya. Master membuat input noisy deterministik `input/anime-before.png`. Masukkan search window NLM berupa bilangan ganjil minimal `7`; tekan Enter untuk memakai nilai default `31`.

Selanjutnya pilih mode:

1. **Sequential** — satu proses di Master.
2. **Local parallel** — beberapa proses di komputer Master.
3. **Distributed** — task dikerjakan Worker yang dipilih.

Untuk mode distributed, Master mencari Worker di jaringan. Pilih Worker yang akan digunakan, scheduler **Dynamic** atau **Static**, dan apakah fault tolerance digunakan. Jika pencarian otomatis tidak menemukan Worker, masukkan alamat IP Worker beserta port jika perlu, misalnya `192.168.1.10` atau `192.168.1.10:5000`.

- **Dynamic:** slot Worker yang lebih dulu selesai menerima task berikutnya.
- **Static:** pembagian task ke slot ditetapkan di awal; tiap slot mengerjakan satu task pada satu waktu.
- **Fault tolerance Yes:** task gagal dicoba kembali pada Worker lain atau slot yang tersedia.
- **Fault tolerance No:** task gagal tidak dicoba kembali dan wilayah hasil yang hilang ditandai magenta.

Hasil denoising disimpan di `output/<nama>-after-nlm-sw<ukuran>.png`, contohnya `output/anime-after-nlm-sw31.png`. Log Worker menampilkan nomor task dan Master pengirimnya. Master menyimpan ringkasan run di folder `logs/`.

## Jika Worker tidak ditemukan

Pastikan Worker sedang berjalan, Master dan Worker berada pada jaringan yang sama, dan firewall Worker mengizinkan TCP `5000` serta UDP `5001`. Jika discovery UDP diblokir oleh jaringan, masukkan alamat Worker secara manual ketika diminta Master.
