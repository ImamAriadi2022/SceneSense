# COLAB_PLAN.md - Rencana Integrasi Google Colab & Peningkatan Robustness Pipeline

Dokumen ini berisi rencana teknis untuk mengintegrasikan project **SceneSense** dengan Google Colab, memperbaiki beberapa bug krusial pada alur *resume training*, dan menjaga alur kerja lokal tetap berjalan tanpa mengubah *command-line interface* (CLI) yang sudah ada.

---

## 1. Perubahan Yang Diperlukan & Alasan Teknis

### A. Override Konfigurasi Path Menggunakan Environment Variables
* **Berkas**: `src/config.py`
* **Perubahan**: Mengubah nilai bawaan (*default value*) field pada `Config` class agar membaca dari Environment Variables jika didefinisikan (misalnya `os.getenv("SCENESENSE_MODEL_PATH", "saved_model")`).
* **Alasan Teknis**: 
  - Notebook Google Colab dilarang memodifikasi argumen CLI `train.py`.
  - Menggunakan Environment Variables memungkinkan kita membelokkan seluruh output (*checkpoint*, model, log) ke Google Drive (`/content/drive/MyDrive/...`) secara dinamis saat di Colab, sementara di lokal tetap menggunakan nilai bawaan folder lokal (`saved_model/`, `logs/`, dll.).

### B. Penambahan Parameter CHECKPOINT_PATH dan LOG_PATH pada Config
* **Berkas**: `src/config.py`, `src/callbacks.py`, `src/trainer.py`
* **Perubahan**: 
  - Mendefinisikan `CHECKPOINT_PATH` (default `"checkpoints"`) dan `LOG_PATH` (default `"logs"`).
  - Menyimpan checkpoint Keras (`latest_checkpoint.keras` dan `best_model.keras`) ke folder `CHECKPOINT_PATH`.
* **Alasan Teknis**: Struktur Google Drive yang diinginkan memisahkan folder `checkpoints/`, `logs/`, dan `saved_model/`. Pada kode asli, checkpoint disimpan di dalam folder `MODEL_PATH` sehingga bercampur dengan berkas *SavedModel* hasil ekspor.

### C. Penambahan Callback TensorBoard
* **Berkas**: `src/callbacks.py`
* **Perubahan**: Menambahkan `tf.keras.callbacks.TensorBoard` ke dalam list callback dengan `log_dir` mengarah ke `LOG_PATH`.
* **Alasan Teknis**: Kebutuhan fungsional agar TensorBoard log tersimpan di Google Drive. Pada kode asli, callback TensorBoard belum diimplementasikan.

### D. Perbaikan Bug Resume Epoch & Hardcoded Path
* **Berkas**: `src/trainer.py`
* **Perubahan**:
  - Mengubah pencarian berkas CSV log di `_get_last_epoch()` agar membaca dari `self.config.LOG_PATH`, bukan nilai hardcoded `"logs"`.
  - Mengubah nilai return `_get_last_epoch()` menjadi `last_epoch + 1` agar training dilanjutkan dari epoch berikutnya, serta membolehkan resume jika hanya 1 epoch selesai (`initial_epoch > 0`).
* **Alasan Teknis**: 
  - Jika log dipindah ke Google Drive, pencarian di `"logs"` lokal akan mengembalikan 0 dan gagal resume.
  - Keras `CSVLogger` mencatat epoch berbasis indeks 0. Jika epoch `N` selesai dicatat, training berikutnya harus mulai dari epoch `N + 1`. Jika tidak ditambah 1, epoch `N` akan dilatih ulang dan ditimpa di log.

### E. Perbaikan Bug Reassignment Model pada Resume Training
* **Berkas**: `src/trainer.py` dan `train.py`
* **Perubahan**:
  - Mengubah method signature `Trainer.train` agar mengembalikan tuple `Tuple[tf.keras.Model, tf.keras.callbacks.History]`.
  - Mengubah pemanggilan di `train.py` menjadi `model, history = trainer.train(model, train_ds, val_ds)`.
* **Alasan Teknis**: Python melewatkan parameter objek secara referensi. Namun, operasi `model = tf.keras.models.load_model(...)` di dalam method `train` meredefinisikan variabel lokal `model` saja. Akibatnya, variabel `model` milik caller (`train.py`) tetap merujuk ke model acak (*scratch*) sebelum dilatih. Hal ini menyebabkan evaluasi dan ekspor model di `train.py` menggunakan model kosong.

### F. Pengecekan Eksistensi Split Dataset
* **Berkas**: `train.py`
* **Perubahan**: Menambahkan pengecekan apakah subfolder `train`, `val`, dan `test` sudah ada di dalam `DATASET_PATH`. Jika sudah ada, lewati pemanggilan `dataset.split_and_save()`.
* **Alasan Teknis**: Proses pembagian dataset memerlukan operasi baca/tulis disk yang cukup berat. Di Google Colab, melewati langkah ini jika dataset sudah terbagi akan menghemat waktu secara signifikan saat notebook dijalankan ulang.

---

## 2. Bagian Yang Dipertahankan (TIDAK BOLEH Diubah)

- **Arsitektur Model CNN**: Kelas `SceneSenseModel` di `src/model.py` tidak boleh diubah susunan layernya agar fungsionalitas utama model tetap konsisten.
- **Pipeline Augmentasi**: `AugmentationPipeline` tetap dipertahankan seperti semula.
- **Argumen CLI**: Antarmuka parameter CLI untuk `train.py`, `inference.py`, dan `download_dataset.py` tetap menggunakan opsi asli tanpa menambahkan opsi baru.
- **Logika Konversi Exporter**: Proses ekspor ke SavedModel, TFLite, dan TFJS tetap menggunakan modul `src/exporter.py` asli.

---

## 3. Risiko Perubahan & Mitigasi

| Perubahan | Risiko Teknis | Mitigasi |
| :--- | :--- | :--- |
| Membaca path dari Environment Variables | Pengguna lokal kebingungan jika path berubah tiba-tiba saat env var tidak sengaja terdefinisi. | Menggunakan prefix yang spesifik (`SCENESENSE_`) dan selalu menyediakan nilai default lokal yang sama persis dengan kode asli jika env var kosong. |
| Mengubah return `Trainer.train` menjadi tuple | Potensi merusak kode eksternal atau notebook lama (`notebook.ipynb`) jika memanggil fungsi tersebut tanpa menampung return model baru. | Menyesuaikan pemanggilan `Trainer.train` di dalam `notebook.ipynb` agar juga menampung model baru secara kompatibel. |
| Memindahkan checkpoint ke `CHECKPOINT_PATH` | Jika pengguna melanjutkan training lama, checkpoint di `saved_model/` tidak terdeteksi di folder baru. | Tambahkan fallback pengecekan di `Trainer.train`: jika checkpoint tidak ditemukan di `CHECKPOINT_PATH` baru, periksa apakah ada di folder `MODEL_PATH` lama sebelum memulai dari scratch. |
