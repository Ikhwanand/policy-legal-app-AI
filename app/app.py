import os 
from datetime import datetime
from typing import Dict, List, Optional

import requests
import streamlit as st 
from dotenv import load_dotenv
import pandas as pd 
import altair as alt

from utils.report import  make_pdf_report

load_dotenv()

st.set_page_config(page_title="AI Legal/Policy Agent", layout="wide")

API_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")


def init_state():
    defaults = {
        "auth_token": None,
        "current_user": None,
        "last_question": "",
        "last_answer": "",
        "last_mode": "",
        "last_hits": [],
        "last_classification": None,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
        
        

init_state()

def logout():
    st.session_state.auth_token = None 
    st.session_state.current_user = None 
    st.session_state.last_question = ""
    st.session_state.last_answer = ""
    st.session_state.last_mode = ""
    st.session_state.last_hits = []
    st.session_state.last_classification = None 
    st.rerun()
    
    

def auth_headers(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    headers = dict(extra or {})
    if st.session_state.auth_token:
        headers["Authorization"] = f"Bearer {st.session_state.auth_token}"
    return headers


def handle_response(response: requests.Response) -> Optional[Dict]:
    if response.status_code == 401:
        st.warning("Sesi berakhir. Silahkan login kembali.")
        logout()
        return None 
    if response.status_code == 403:
        st.error("Akses ditolak.")
        return None 
    if not response.ok:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text

        st.error(f"Permintaan gagal ({response.status_code}): {detail}")
        return None 
    try:
        return response.json()
    except ValueError:
        st.error("Respons backend bukan JSON.")
        return None
    


def api_get(path: str, timeout: int = 30) -> Optional[Dict]:
    try:
        response = requests.get(f"{API_URL}{path}", headers=auth_headers(), timeout=timeout)
    except requests.RequestException as exc:
        st.error(f"Gagal menghubungi backend: {exc}")
        return None 
    return handle_response(response) 


def api_post_json(path: str, payload: Dict, timeout: int =120) -> Optional[Dict]:
    try:
        response = requests.post(
            f"{API_URL}{path}",
            headers=auth_headers({"Content-Type": "application/json"}),
            json=payload,
            timeout=timeout,
        )   
    except requests.RequestException as exc:
        st.error(f"Gagal menghubungi backend: {exc}")
        return None 
    return handle_response(response)



def api_post_files(path: str, files, timeout: int = 300) -> Optional[Dict]:
    try:
        response = requests.post(
            f"{API_URL}{path}",
            headers=auth_headers(),
            files=files,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        st.error(f"Gagal menghubungi backend: {exc}")
        return None 
    return handle_response(response)


def fetch_current_user():
    profile = api_get("/auth/me")
    if profile:
        st.session_state.current_user = profile 
        
        
def login_form():
    st.title("AI Legal/Policy Agent - Login")
    tab_login, tab_register = st.tabs(['Login', 'Register'])
    
    with tab_login:
        with st.form("login-form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Login")
        if submitted:
            if not username or not password:
                st.warning("Lengkapi username dan password.")
                return 
            token_payload = api_post_json("/auth/login", {"username": username, "password": password})
            if token_payload:
                st.session_state.auth_token = token_payload["access_token"]
                fetch_current_user()
                st.success("Login berhasil.")
                st.rerun()
    
    with tab_register:
        with st.form("register-form"):
            reg_username = st.text_input("Username")
            reg_email = st.text_input("Email")
            reg_password = st.text_input("Password", type="password")
            reg_submit = st.form_submit_button("Daftar")
        if reg_submit:
            if not reg_username or not reg_email or not reg_password:
                st.warning("Semua kolom wajib diisi.")
            else:
                result = api_post_json(
                    "/auth/register",
                    {"username": reg_username, "email": reg_email, "password": reg_password},
                )
                if result:
                    st.success("Registrasi berhasil. Silahkan login.")
                    

def format_timestamp(value: str) -> str:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%d %b %Y %H:%M")
    except ValueError:
        return value
    

def describe_structure(hit: Dict) -> str:
    parts: List[str] = []
    if hit.get("bab"):
        title = f"Bab {hit['bab']}"
        if hit.get("bab_title"):
            title += f" - {hit['bab_title']}"
        parts.append(title)
    if hit.get("bagian"):
        bagian = f"Bagian {hit['bagian']}"
        if hit.get("bagian_title"):
            bagian += f" - {hit['bagian_title']}"
        parts.append(bagian)
    if hit.get("paragraf"):
        paragraf = f"Paragraf {hit['paragraf']}"
        if hit.get("paragraf_title"):
            paragraf += f" - {hit['paragraf_title']}"
        parts.append(paragraf)
    if hit.get("pasal"):
        pasal = f"Pasal {hit['pasal']}"
        if hit.get("ayat"):
            pasal += f" ayat ({hit['ayat']})"
        parts.append(pasal)
    elif hit.get("ayat"):
        parts.append(f"Ayat ({hit['ayat']})")
    return " | ".join(parts)
    

def render_scrape_panel():
    st.caption("Scrape dokumen dari peraturan.bpk.go.id dan langsung masukkan ke knowledge base.")
    with st.form("scrape-form"):
        keyword = st.text_input("Kata kunci peraturan", placeholder="contoh: retribusi parkir")
        tentang = st.text_input("Tentang (opsional)", placeholder="contoh: pajak daerah")
        nomor = st.text_input("Nomor peraturan (opsional)", placeholder="contoh: 12/2023")
        max_docs = st.slider("Jumlah hasil peraturan", 1, 5, 2)
        downloads_per_doc = st.slider("File per hasil", 1, 5, 1)
        auto_ingest = st.checkbox("Otomatis ingest & embedding", value=True)
        scrape_submit = st.form_submit_button("Cari & proses")
    if scrape_submit:
        if not keyword:
            st.warning("Masukkan kata kunci terlebih dahulu.")
            return
        payload = {
            "keyword": keyword,
            "tentang": tentang or None,
            "nomor": nomor or None,
            "max_documents": max_docs,
            "downloads_per_document": downloads_per_doc,
            "auto_ingest": auto_ingest,
        }
        result = api_post_json("/admin/scrape", payload, timeout=600)
        if result:
            docs = result.get("documents", [])
            st.success(f"Berhasil memproses {len(docs)} peraturan.")
            for item in docs:
                title = item.get("title") or "Peraturan"
                subjects = item.get("subjects") or []
                with st.expander(title):
                    if item.get("description"):
                        st.write(item["description"])
                    if subjects:
                        st.caption(", ".join(subjects))
                    downloads = item.get("downloaded_files") or []
                    if not downloads:
                        st.info("Tidak ada file yang dipilih (cek filter/ekstensi).")
                        continue
                    for download in downloads:
                        status = (
                            f"tersimpan (doc #{download.get('document_id')})"
                            if download.get("document_id")
                            else "belum diunggah"
                        )
                        chunk_info = (
                            f", chunk {download.get('chunks_indexed')}"
                            if download.get("chunks_indexed")
                            else ""
                        )
                        st.markdown(
                            f"- [{download.get('filename')}]({download.get('url')}) — {status}{chunk_info}"
                        )


def render_admin_dashboard():
    st.subheader("Kelola Knowledge Base")
    tab_upload, tab_scrape = st.tabs(["Upload Dokumen", "Scrape JDIH BPK"])
    
    with tab_upload:
        uploaded_files = st.file_uploader(
            "Unggah dokumen hukum (PDF/DOCX)",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key="admin-uploader",
        )
        if st.button("Unggah ke Knowledge Base", key="upload-btn"):
            if not uploaded_files:
                st.warning("Pilih minimal satu dokumen.")
            else:
                files_payload = [
                    ("files", (file.name, file.getvalue(), file.type or "application/octet-stream"))
                    for file in uploaded_files
                ]
                result = api_post_files("/admin/upload", files_payload)
                if result:
                    st.success(f"Berhasil mengunggah {len(result)} dokumen.")
    
    with tab_scrape:
        render_scrape_panel()
    
    documents = api_get("/admin/documents") or []
    if documents:
        df = pd.DataFrame(documents)
        df["uploaded_at"] = pd.to_datetime(df["uploaded_at"], utc=True).dt.tz_convert(None)

        cutoff = pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=7)
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Dokumen", len(df))
        col2.metric("Total Uploader", df["uploaded_by"].nunique())
        col3.metric("Upload (7 hari)", df[df["uploaded_at"] >= cutoff].shape[0])

        daily = (
            df.groupby(df["uploaded_at"].dt.date)["id"]
            .count()
            .reset_index(name="jumlah_dokumen")
        )
        st.bar_chart(daily.set_index("uploaded_at"))
        
        
        rows = [
            {
                "Nama Dokumen": doc["original_filename"],
                "Uploader": doc.get("uploader_username") or "-",
                "Tanggal": format_timestamp(doc["uploaded_at"]),
                "Vector Chunks": doc.get("chunks_indexed", "-"),
                "Sumber": (doc.get("metadata") or [{}])[0].get("source", "-"),
                "Metadata": (doc.get("metadata") or [{}])[0].get("data", {}).get("keyword", "-"),
            }
            for doc in documents
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)
    else:
        st.info("Belum ada dokumen di knowledge base.")
        
    
def render_history_panel():
    st.subheader("Riwayat Analisis & Insight")
    history = api_get("/analyses/history") or []
    if not history:
        st.info("Belum ada riwayat konsultasi.")
        return
    df = pd.DataFrame(history)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True).dt.tz_convert(None)
    st.dataframe(
        df[["question", "classification_label", "created_at"]],
        use_container_width=True,
        hide_index=True,
    )
    
    if df["classification_label"].notna().any():
        agg = (
            df[df["classification_label"].notna()]
            .groupby("classification_label")["id"]
            .count()
            .reset_index(name="jumlah")
        )
        chart = (
            alt.Chart(agg)
            .mark_bar()
            .encode(x="classification_label", y="jumlah", tooltip=["classification_label", "jumlah"])
            .properties(height=300)
        )
        st.altair_chart(chart, use_container_width=True)
    
    st.markdown("#### Detail terbaru")
    for record in history[:5]:
        created = format_timestamp(record["created_at"])
        header = f"{created} · {record['question'][:80]}"
        with st.expander(header):
            st.markdown(record["answer"])
            if record.get("classification_label"):
                st.caption(
                    f"Kategori: **{record['classification_label']}** "
                    f"(skor ~{(record.get('classification_score') or 0):.2f})"
                )
            if st.button("Muat jawaban ini", key=f"use-history-{record['id']}"):
                st.session_state.last_question = record["question"]
                st.session_state.last_answer = record["answer"]
                st.session_state.last_mode = record["mode"]
                st.session_state.last_hits = record.get("contexts") or []
                if record.get("classification_label"):
                    st.session_state.last_classification = {
                        "label": record["classification_label"],
                        "score": record.get("classification_score") or 0.0,
                    }
                else:
                    st.session_state.last_classification = None
                st.success("Riwayat dimuat ke panel jawaban.")
    
    

def render_chat_interface(use_llm: bool, top_k: int):
    st.subheader("Konsultasi AI")
    with st.form("chat-form"):
        question = st.text_input(
            "Ajukan pertanyaan kebihakan / isu",
            placeholder="Contoh: Apa dasar hukum retribusi parkir di kabupaten?",
        )
        submitted = st.form_submit_button("Kirim pertanyaan")
    if submitted:
        if not question:
            st.warning("Pertanyaan tidak boleh kosong.")
        else:
            response = api_post_json("/chat/ask", {"question": question, "use_llm": use_llm, "top_k": top_k})
            if response:
                st.session_state.last_question = question
                st.session_state.last_answer = response.get("answer", "")
                st.session_state.last_mode = response.get("mode", "")
                st.session_state.last_hits = response.get("context", [])
                st.session_state.last_classification = response.get("classification", None)
    
    if st.session_state.last_answer:
        if st.session_state.last_mode == "llm_fallback":
            st.info("Mode LLM tidak tersedia. Menampilkan ringkasan konteks sebagai fallback.")
        st.subheader("Jawaban")
        st.markdown(st.session_state.last_answer)
        
        if st.session_state.last_classification:
            st.caption(
                f"Kategori dominan: **{st.session_state.last_classification['label']} "
                f"(skor ~{st.session_state.last_classification['score']:.2f})"
            )
        
        with st.expander("Konteks (Top-K)"):
            for idx, hit in enumerate(st.session_state.last_hits, start=1):
                location_bits = []
                if hit.get("page"):
                    location_bits.append(f"hal. {hit['page']}")
                if hit.get("section"):
                    location_bits.append(f"paragraf {hit['section']}")
                if hit.get("section_chunk"):
                    location_bits.append(f"bagian {hit['section_chunk']}")
                location = f" ({', '.join(location_bits)})" if location_bits else ""
                score_text = f" - skor {hit['score']:.3f}" if hit.get("score") is not None else ""
                source = hit.get("doc_id") or hit.get("source", "-")
                st.markdown(f"**{idx}. {source}**{location}{score_text}")
                st.write(hit.get("text", "")[:1000])
                structure = describe_structure(hit)
                if structure:
                    st.caption(structure)
                
        
        pdf_bytes = make_pdf_report(
            st.session_state.last_question,
            st.session_state.last_answer,
            st.session_state.last_hits,
        )
        st.download_button(
            "Download Laporan (PDF)",
            data=pdf_bytes,
            file_name="laporan_rekomendasi.pdf",
            mime="application/pdf",
        )
        
        

def render_dashboard():
    user = st.session_state.current_user or {}
    st.sidebar.header("Konfigurasi")
    use_llm_val = st.sidebar.toggle("Use LLM (Gemini)", value=st.session_state.get("use_llm", False), key="use_llm")
    top_k_val = st.sidebar.slider("Top-K Context", 1, 10, st.session_state.get("top_k", 5), key="top_k")
    st.sidebar.caption("Login -> Upload (admin) -> Ajukan pertanyaan.")
    
    st.sidebar.markdown('---')
    st.sidebar.write(f"Masuk sebagai: **{user.get('username', '-')}** (`{user.get('role', '-')}`)")
    if st.sidebar.button("Keluar"):
        logout()
        
        
    st.title("AI Legal/Policy Agent - Dashboard")
    st.write("Prototype untuk analisis regulasi dan rekomendasi kebijakan berbasis dokumen.")
    
    if user.get("role") == "admin":
        render_admin_dashboard()
        st.markdown("---")
        
    tab_chat, tab_history = st.tabs(["AI Konsultasi", "Riwayat & Insight"])
    with tab_chat:
        render_chat_interface(use_llm_val, top_k_val)
    with tab_history:
        render_history_panel()
    
    

if not st.session_state.auth_token or not st.session_state.current_user:
    login_form()
else:
    render_dashboard()
    
    
st.caption("Prototype Magang - Pemerintah Kabupaten Sumbawa")
