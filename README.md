# Distributed Image Upscaler

Master–Worker image processing with TCP, automatic UDP worker discovery, OpenCV, NumPy, and Pillow. The Master coordinates and stitches distributed tasks; each Worker uses a persistent process pool.

## Setup

On each machine, install Python dependencies from `requirements.txt`. On NixOS, `nix develop` provides the development environment.

Start one Worker on each worker computer (allow UDP 5001 and TCP 5000 through its firewall):

```sh
python worker.py
```

Worker otomatis bind ke semua antarmuka jaringan, memakai port default, dan membuat process pool sebesar jumlah CPU core yang terdeteksi. Opsi konfigurasi tetap tersedia bila diperlukan.

The Worker announces itself over UDP discovery port 5001. The Master and Workers need LAN broadcast connectivity. If discovery is blocked, pass workers manually with `--worker IP[:PORT]`.

## Interactive Master (recommended)

```sh
python master.py
```

The terminal scans for Workers, displays their names, addresses, and process slots, then asks for the input image, scale, execution mode, Worker selection, and scheduler. The modes are sequential baseline, local parallel, and distributed. Distributed mode offers dynamic or static scheduling and optional fault simulation. Results are saved under `after/`; timing, throughput, image checksum, task progress, and recovery messages appear in the terminal.

For unattended runs, the same choices remain available as command-line options. `--workers` limits detected nodes for the 1/2/4/6/9 Worker scalability scenarios; `--rows-per-task` controls strip granularity.

For a fault injection run, `--fail-after N` simulates losing the Worker that completed task N by removing it from scheduling. A task whose connection fails is retried on another available Worker. For a real failure demonstration, stop a Worker process during a run and observe retry behavior.

## UAS evaluation

Run sequential, local-parallel, distributed, static-versus-dynamic, fault-recovery, and 1/2/4/6/9 Worker scenarios on the same square input sizes. Repeat each run and retain terminal logs and Worker-side evidence. The required 1 Master + 9 Worker physical-node demonstration must be recorded on physical computers; local simulations do not establish that requirement.
