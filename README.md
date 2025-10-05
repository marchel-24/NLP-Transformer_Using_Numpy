# NLP-Transformer_Using_Numpy

**Nama:** `Marchel Rianra Glendrikho Simanjuntak`

**NIM:** `22/494013/TK/54157`

[![Language](https://img.shields.io/badge/Language-Python-blue.svg)](https://www.python.org/)
[![Library](https://img.shields.io/badge/Library-NumPy-orange.svg)](https://numpy.org/)
[![Visualization](https://img.shields.io/badge/Feature-Attention%20Visualization-green.svg)]()
[![Positional Encoding](https://img.shields.io/badge/Feature-RoPE%20Supported-blueviolet.svg)]()

## 📖 Tentang Proyek Ini

Proyek ini merupakan implementasi arsitektur **Decoder-Only Transformer (GPT-style)** dari nol (*from scratch*), menggunakan **NumPy** sebagai dasar perhitungan numerik.  
Seluruh komponen utama Transformer dibangun manual tanpa bantuan framework seperti PyTorch atau TensorFlow.

Fokus utama implementasi ini adalah membangun alur *forward pass* dari Transformer, mulai dari *embedding* hingga menghasilkan distribusi probabilitas token berikutnya.

Selain fitur dasar, model juga dilengkapi beberapa fitur lanjutan:
- 🔗 **Weight Tying** antara embedding input dan proyeksi output.
- 🎨 **Visualisasi Distribusi Attention** menggunakan Matplotlib.
- 🌀 **Alternatif Positional Encoding (RoPE – Rotary Positional Encoding)**.

---

## 🏗️ Arsitektur & Komponen

Proyek ini mencakup komponen-komponen utama berikut:

| Komponen | Deskripsi |
|-----------|------------|
| **Token Embedding** | Mengubah token menjadi vektor berdimensi *d_model* menggunakan matriks embedding. |
| **Positional Encoding** | Menambahkan informasi posisi token. Mendukung dua jenis encoding: Sinusoidal klasik dan Rotary Positional Encoding (RoPE). |
| **Causal Masking** | Memastikan model hanya memperhatikan token masa lalu (digunakan dalam decoder). |
| **Scaled Dot-Product Attention** | Menghitung perhatian antar token dengan operasi dot product yang dinormalisasi oleh `√d_k`. |
| **Multi-Head Attention** | Menggabungkan beberapa head attention secara paralel dan mengembalikannya ke dimensi awal. |
| **Feed Forward Network (FFN)** | Dua lapisan linier dengan fungsi aktivasi ReLU. |
| **Layer Normalization** | Normalisasi fitur per token untuk stabilitas pelatihan. |
| **Decoder Block** | Kombinasi Multi-Head Attention, Feed Forward, residual connection, dan layer normalization. |
| **Weight Tying** | Menyamakan bobot antara embedding input dan output projection (`W_out = E_in^T`). |
| **Attention Visualization** | Menampilkan heatmap distribusi perhatian antar token. |
| **Rotary Positional Encoding (RoPE)** | Alternatif dari sinusoidal encoding yang memperkenalkan posisi **relatif** antar token melalui rotasi vektor embedding. Dapat diaktifkan dengan parameter `use_rope=True`. |


## 📖References
1. Vaswani, A., Shazeer, N., Parmar, N., et al. *“Attention Is All You Need.”*  
   In: Proceedings of the 31st Conference on Neural Information Processing Systems (NIPS 2017), pp. 5998–6008.  
   [IEEE Xplore](https://ieeexplore.ieee.org/document/9414641)

2. **Roh, Dorsa.** *“Transformer from Scratch.”*  
   GitHub repository: [https://github.com/DorsaRoh/transformer-from-scratch](https://github.com/DorsaRoh/transformer-from-scratch)
