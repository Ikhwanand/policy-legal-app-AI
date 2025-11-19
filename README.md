# AI Legal/Policy Agent - Interactive Dashboard (Prototype)

Prototype dashboard untuk Pemerintah Kabupaten Sumbawa yang membaca dokumen hukum (PDF/DOCX), melakukan pencarian semantik, menjawab isu kebijakan, mengklasifikasikan dampak, serta menghasilkan laporan yang dapat diunduh. Seluruh alur berjalan lokal dan memanfaatkan Agno Agent untuk reasoning generatif.

## Setup
```bash
python -m venv .venv
source venv/Scripts/activate
# Windows: .venv\Scripts\activate | Mac/Linux: source .venv/bin/activate

pip install -r requirements.txt

# Jalankan API (posisi folder root, bukan di dalam app/)
uvicorn app.api:app --reload

# Jalankan dashboard Streamlit
streamlit run app/app.py
```

## Notes
- Format yang didukung: `.pdf` dan `.docx`.
- Model embedding akan otomatis diunduh saat pertama kali dijalankan (butuh koneksi internet).
- File unggahan, index FAISS, dan database SQLite tersimpan di `app/storage/`.
- Kredensial admin default diatur lewat `.env` (`DEFAULT_ADMIN_USERNAME`, `DEFAULT_ADMIN_PASSWORD`, `DEFAULT_ADMIN_EMAIL`).
- Dataset contoh untuk klasifikasi isu berada di `app/models/impact_training_samples.json` dan dipakai otomatis ketika model belum tersedia.

## Fitur SIKAP 2025
1. **Ekstraksi struktur regulasi** di `app/nlp/ingest.py` menandai Bab/Bagian/Paragraf/Pasal/Ayat secara rule-based sehingga referensi pasal muncul di dashboard dan laporan PDF.
2. **Scraping JDIH BPK** langsung dari dashboard admin. Cukup masukkan kata kunci, jumlah hasil, dan jumlah file. Sistem mengunduh PDF/DOCX dari https://peraturan.bpk.go.id, menyimpannya ke storage, lalu mengindeks ke FAISS.
3. **Riwayat & insight** menyimpan setiap konsultasi dalam tabel `analyses` lengkap dengan konteks dan skor klasifikasi. Tab khusus menampilkan tabel kronologis, chart distribusi isu, serta tombol untuk memuat ulang jawaban sebelumnya.
4. **Klasifikasi dengan optimasi** memakai grid-search (Random Forest vs. SVM) + cross-validation untuk memilih model terbaik sebelum disimpan ke `app/storage/impact_classifier.pkl`.
5. **Agno RAG Agent** tetap menjadi core reasoning (menggunakan model NVIDIA / Gemini) dengan fallback extractive serta konteks beranotasi struktur hukum.
6. **Pelaporan otomatis**: tombol unduh PDF menyertakan Bab/Pasal/Ayat pada setiap referensi konteks.

## Scraping Peraturan BPK
1. Login sebagai admin di Streamlit.
2. Buka tab **Scrape JDIH BPK**, isi kata kunci pencarian, jumlah peraturan yang ingin diambil, dan jumlah file per peraturan.
3. Aktifkan opsi *Otomatis ingest & embedding* bila ingin langsung dimasukkan ke knowledge base.
4. Tekan **Cari & proses**. Hasil scrape akan menampilkan daftar file beserta status penyimpanan dan jumlah chunk vektor.
5. Dokumen yang berhasil otomatis muncul pada tabel dokumen admin dan siap dipakai untuk konsultasi.

## Riwayat & Insight
- Tab **Riwayat & Insight** menampilkan tabel pertanyaan terbaru, grafik distribusi kategori impact, serta daftar detail rekomendasi.
- Tombol **Muat jawaban ini** pada setiap histori akan mengisi ulang panel konsultasi beserta konteks, sehingga pimpinan dapat meninjau/mengunduh ulang tanpa menunggu inferensi baru.

## Future Development
- Skor dampak kuantitatif dan reasoning multi-skenario (mis. best case vs worst case).
- Ekspor laporan multi-format (PDF + DOCX) dengan lampiran pasal.
- Active learning UI agar admin dapat memberi label ulang hasil klasifikasi dan melakukan retraining langsung dari dashboard.
- Integrasi Role-Based Access Control (RBAC) untuk membedakan hak akses admin vs pimpinan OPD.

## Workflow Explanation

### Upload & Indexing
1. **Upload Document**: pengguna mengunggah PDF/DOCX atau menarik dari JDIH BPK.
2. **Save to Folder**: file disimpan ke `app/storage/uploads`.
3. **Ingest & Structuring**: teks diekstraksi lalu parser mendeteksi Bab/Pasal/Ayat.
4. **Embedding**: chunk teks diubah menjadi vektor (Sentence Transformers).
5. **Vector DB (FAISS)**: embedding + metadata disimpan untuk pencarian semantik.

### Query & Answering
1. **User Query**: pengguna memasukkan pertanyaan pada dashboard.
2. **Query Embedding**: pertanyaan di-embedding dengan model yang sama.
3. **Semantic Search**: FAISS mencari konteks Top-K yang paling relevan.
4. **QA Agent**:
   - Mode 1: Extractive (tanpa LLM, gratis).
   - Mode 2: Generative (Agno Agent + NVIDIA/Gemini) dengan reasoning dan referensi.
5. **Display Answer**: jawaban + referensi pasal ditampilkan, bisa diunduh sebagai PDF.

## Version 1 - With LLM (RAG + Generative Answer)

```mermaid
flowchart TD
    subgraph Upload_&_Indexing
        A[Upload Document (PDF/DOCX) - Streamlit]
        B[Save to storage/uploads (pathlib)]
        C[Ingest & Structuring (app/nlp/ingest.py)]
        D[Embedding (Sentence Transformers)]
        E[(FAISS Index + Metadata) storage/]
    end

    subgraph Query_&_Answer
        F[User Query (Streamlit input)]
        G[Query Embedding (same model)]
        H[Semantic Search Top-K contexts]
        I[QA Agent (app/agent/qa_agent.py)\n+ Prompt = Query + Contexts]
        J[LLM Generation (Agno + NVIDIA/Gemini)]
        K[Answer + References (Dashboard)]
        L[[Download Report]]
    end

    A-->B-->C-->D-->E
    F-->G-->H-->I-->J-->K-->L
    E-->H
```

## Version 2 - Without LLM (Pure Extractive)

```mermaid
flowchart TD
    subgraph Upload_&_Indexing
        A[Upload Document]
        B[Save to storage/uploads]
        C[Ingest & Chunking]
        D[Embedding]
        E[(FAISS Index)]
    end

    subgraph Query_&_Answer
        F[User Query]
        G[Query Embedding]
        H[Semantic Search Top-K]
        I[Extractive Answer\n(summarize context + quote)]
        J[Result + Source (Dashboard)]
        K[[Download Report]]
    end

    A-->B-->C-->D-->E
    F-->G-->H-->I-->J-->K
    E-->H
```

## app/ Directory Structure
- `agent/` – modul QA (RAG + Agno) beserta cache jawaban.
- `backend/` – FastAPI (auth, upload, scraping, history, DB models).
- `models/` – pipeline klasifikasi, sample dataset, dan penyimpanan model.
- `nlp/` – ekstraksi dokumen, chunking, embedding, dan scraper JDIH BPK.
- `storage/` – database SQLite, FAISS index, model terlatih, dan file unggahan.
- `utils/` – utilitas laporan PDF/markdown.
- `app.py` – entry point Streamlit (UI dashboard & insight).
