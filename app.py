import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import base64
import os
import tensorflow as tf

st.set_page_config(layout="wide", page_title="Botani Vision")

CLASS_NAMES = ["sakit", "sehat"]
IMG_SIZE    = (224, 224)

# ── Load model TFLite (bukan Keras) ───────────────────────────
@st.cache_resource
def load_model():
    # tf.lite.Interpreter digunakan khusus untuk file .tflite
    interpreter = tf.lite.Interpreter(model_path="botani_model.tflite")
    # allocate_tensors() wajib dipanggil sebelum inferensi
    interpreter.allocate_tensors()
    return interpreter

model = load_model()

def get_base64_image(image_filename):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    image_path  = os.path.join(current_dir, image_filename)
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""

if "page" not in st.session_state:
    st.session_state.page = "landing"

# ============================================================
# HALAMAN 1: LANDING
# ============================================================
if st.session_state.page == "landing":

    bg_base64 = get_base64_image("bground.jpeg")

    st.markdown("""
    <style>
    #MainMenu, footer, header { visibility: hidden; }
    .block-container { padding: 0 !important; max-width: 100% !important; }
    [data-testid="stAppViewContainer"] { padding: 0 !important; }
    div[data-testid="stButton"] > button {
        background: linear-gradient(135deg, #27913e, #1a6630) !important;
        color: #ffffff !important;
        border: 2px solid rgba(255,255,255,0.30) !important;
        border-radius: 60px !important;
        font-size: 18px !important;
        font-weight: 700 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        padding: 18px 56px !important;
        box-shadow: 0 8px 30px rgba(0,0,0,0.35) !important;
        transition: all 0.3s ease !important;
        width: 320px !important;
        height: 64px !important;
        display: block !important;
        margin: 0 auto !important;
    }
    div[data-testid="stButton"] > button:hover {
        background: linear-gradient(135deg, #2daa48, #1e7a34) !important;
        transform: translateY(-3px) !important;
        box-shadow: 0 14px 40px rgba(0,0,0,0.45) !important;
    }
    </style>
    """, unsafe_allow_html=True)

    landing_html = f"""
    <!DOCTYPE html><html lang="id"><head><meta charset="UTF-8">
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Lato:wght@300;400;700&display=swap" rel="stylesheet">
    <style>
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    html {{ scroll-behavior:smooth; }}
    body {{
        font-family: 'Lato', sans-serif;
        background: linear-gradient(rgba(10,40,10,0.60), rgba(10,40,10,0.60)),
                    url('data:image/jpeg;base64,{bg_base64}');
        background-size: cover; background-attachment: fixed; background-position: center;
        overflow-x: hidden;
    }}
    .slide {{ min-height:100vh; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; padding:60px 30px; }}
    .title {{ font-family:'Playfair Display',serif; font-size:100px; font-weight:900; color:#ffffff; letter-spacing:8px; text-transform:uppercase; text-shadow:0 4px 30px rgba(0,0,0,0.6); margin-bottom:20px; line-height:1.05; }}
    .divider {{ width:90px; height:4px; background:#5cb85c; border-radius:2px; margin:0 auto 28px auto; }}
    .subtitle {{ font-size:22px; color:#ffffff; max-width:720px; margin:0 auto; line-height:1.9; font-weight:300; text-shadow:0 2px 10px rgba(0,0,0,0.5); }}
    .scroll-hint {{ margin-top:70px; font-size:12px; color:rgba(255,255,255,0.65); letter-spacing:4px; text-transform:uppercase; animation:bounce 2s infinite; }}
    .scroll-hint .arrow {{ display:block; font-size:32px; margin-top:8px; color:#5cb85c; }}
    @keyframes bounce {{ 0%,100% {{ transform:translateY(0); }} 50% {{ transform:translateY(10px); }} }}
    .slide2 {{ min-height:100vh; display:flex; flex-direction:column; align-items:center; justify-content:center; padding:80px 40px; }}
    .section-title {{ font-family:'Playfair Display',serif; font-size:44px; color:#ffffff; text-align:center; margin-bottom:14px; letter-spacing:2px; text-shadow:0 2px 14px rgba(0,0,0,0.5); }}
    .section-underline {{ width:60px; height:3px; background:#5cb85c; border-radius:2px; margin:0 auto 50px auto; }}
    .cards-row {{ display:flex; gap:36px; justify-content:center; align-items:stretch; flex-wrap:wrap; max-width:1080px; width:100%; }}
    .card {{ flex:1; min-width:300px; max-width:470px; background:linear-gradient(145deg,#1a6630,#27913e); border-radius:24px; padding:50px 44px; box-shadow:0 16px 48px rgba(0,0,0,0.40); border:1px solid rgba(255,255,255,0.12); position:relative; overflow:hidden; transition:transform 0.35s ease,box-shadow 0.35s ease; }}
    .card:hover {{ transform:translateY(-8px); box-shadow:0 28px 60px rgba(0,0,0,0.50); }}
    .card::before {{ content:''; position:absolute; top:-50px; right:-50px; width:160px; height:160px; border-radius:50%; background:rgba(255,255,255,0.06); pointer-events:none; }}
    .card-badge {{ display:inline-block; background:rgba(255,255,255,0.18); color:#ffffff; font-size:11px; font-weight:700; letter-spacing:3px; text-transform:uppercase; padding:6px 18px; border-radius:40px; margin-bottom:22px; border:1px solid rgba(255,255,255,0.28); }}
    .card h2 {{ font-family:'Playfair Display',serif; font-size:26px; color:#ffffff; margin-bottom:18px; font-weight:700; }}
    .card p {{ font-size:16px; color:#dcedc8; line-height:1.85; font-weight:300; margin-bottom:14px; }}
    .card ol {{ padding-left:0; margin:0; list-style:none; counter-reset:step-counter; }}
    .card ol li {{ font-size:16px; color:#dcedc8; line-height:1.75; font-weight:300; margin-bottom:18px; counter-increment:step-counter; position:relative; padding-left:40px; }}
    .card ol li::before {{ content:counter(step-counter); position:absolute; left:0; top:2px; width:26px; height:26px; background:rgba(255,255,255,0.20); border-radius:50%; font-size:12px; font-weight:700; color:#ffffff; display:flex; align-items:center; justify-content:center; }}
    .card ol li b {{ display:block; color:#a5d6a7; font-weight:700; font-size:15px; margin-bottom:3px; }}
    .hl-green {{ color:#a5d6a7; font-weight:700; }}
    .hl-red {{ color:#ef9a9a; font-weight:700; }}
    .btn-placeholder {{ margin-top:48px; height:80px; width:100%; display:flex; align-items:center; justify-content:center; }}
    .btn-placeholder span {{ font-family:'Lato',sans-serif; font-size:13px; color:rgba(255,255,255,0.4); letter-spacing:2px; }}
    </style></head><body>
    <div class="slide">
        <div class="title">Botani Vision</div>
        <div class="divider"></div>
        <div class="subtitle">Deteksi kesehatan tanaman Anda dengan kecerdasan buatan.<br>Analisis daun secara cepat, akurat, dan otomatis &mdash; hanya dalam hitungan detik.</div>
        <div class="scroll-hint">Gulir ke bawah untuk info lebih lanjut<span class="arrow">&#8595;</span></div>
    </div>
    <div class="slide2">
        <div class="section-title">Kenali Lebih Jauh</div>
        <div class="section-underline"></div>
        <div class="cards-row">
            <div class="card">
                <span class="card-badge">Tentang Kami</span>
                <h2>Tentang Botani Vision</h2>
                <p>Botani Vision adalah platform berbasis kecerdasan buatan yang dirancang khusus untuk mendeteksi kondisi kesehatan daun melalui analisis gambar digital.</p>
                <p>Pengguna cukup mengunggah foto daun, lalu sistem secara otomatis mengidentifikasi apakah daun dalam kondisi <span class="hl-green">sehat</span> atau mengalami <span class="hl-red">penyakit tertentu</span> &mdash; tanpa memerlukan keahlian khusus di bidang botani.</p>
            </div>
            <div class="card">
                <span class="card-badge">Cara Kerja</span>
                <h2>Cara Kerja</h2>
                <ol>
                    <li><b>Upload Foto</b>Unggah foto daun dari perangkat Anda.</li>
                    <li><b>Analisis AI</b>Model AI memproses citra daun secara mendalam menggunakan deep learning.</li>
                    <li><b>Hasil Diagnosis</b>Dapatkan hasil diagnosis lengkap dalam hitungan detik.</li>
                </ol>
            </div>
        </div>
        <div class="btn-placeholder"><span>&#8595; klik tombol di bawah untuk mulai</span></div>
    </div>
    </body></html>
    """

    components.html(landing_html, height=2150, scrolling=True)

    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("Mulai Analisis →", use_container_width=False):
            st.session_state.page = "dashboard"
            st.rerun()

# ============================================================
# HALAMAN 2: DASHBOARD
# ============================================================
elif st.session_state.page == "dashboard":

    bg_base64 = get_base64_image("bground.jpeg")

    st.markdown(f"""
    <style>
    #MainMenu, footer, header {{ visibility: hidden; }}
    .stApp {{
        background: linear-gradient(rgba(10,40,10,0.65), rgba(10,40,10,0.65)),
                    url(data:image/jpeg;base64,{bg_base64});
        background-size: cover; background-position: center; background-attachment: fixed;
    }}
    .block-container {{ max-width:650px !important; margin-top:80px !important; margin-bottom:80px !important; padding:50px !important; background:rgba(20,70,30,0.72); backdrop-filter:blur(20px); border-radius:30px; }}
    .stRadio label, .stRadio p {{ color:#ffffff !important; font-weight:500 !important; }}
    .stRadio > div > label {{ color:#ffffff !important; }}
    [data-testid="stFileUploader"] label,
    [data-testid="stFileUploader"] p,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploader"] small,
    [data-testid="stFileUploaderDropzone"] p,
    [data-testid="stFileUploaderDropzone"] span {{ color:#ffffff !important; }}
    [data-testid="stFileUploaderDropzoneInstructions"] span,
    [data-testid="stFileUploaderDropzoneInstructions"] p {{ color:#ffffff !important; }}
    [data-testid="stFileUploaderDropzone"] {{ border:2px dashed rgba(255,255,255,0.45) !important; border-radius:12px !important; background:rgba(255,255,255,0.07) !important; }}
    [data-testid="stFileUploaderDropzone"] button {{ background:rgba(255,255,255,0.18) !important; color:#ffffff !important; border:1px solid rgba(255,255,255,0.40) !important; border-radius:8px !important; }}
    .stFileUploader > label > div > p {{ color:#ffffff !important; }}
    [data-testid="stFileUploaderDropzone"] section p {{ color:rgba(255,255,255,0.70) !important; }}
    [data-testid="stCameraInput"] label {{ color:#ffffff !important; }}
    .stMarkdown p, .stMarkdown span {{ color:#ffffff !important; }}
    div[data-testid="stButton"] > button {{ background:linear-gradient(135deg,#27913e,#1a6630) !important; color:#ffffff !important; border:none !important; border-radius:12px !important; font-weight:700 !important; letter-spacing:1px !important; font-size:16px !important; padding:12px 28px !important; }}
    div[data-testid="stButton"] > button:hover {{ background:linear-gradient(135deg,#2daa48,#1e7a34) !important; transform:translateY(-2px) !important; }}
    </style>
    """, unsafe_allow_html=True)

    if st.button("← Kembali ke Beranda"):
        st.session_state.page = "landing"
        st.rerun()

    st.markdown('<div style="font-size:52px;color:white;text-align:center;font-weight:800;margin-bottom:8px;">Upload Daun</div>', unsafe_allow_html=True)
    st.markdown('<div style="text-align:center;color:#dfffdc;font-size:16px;margin-bottom:30px;">Analisis AI Botani Vision</div>', unsafe_allow_html=True)

    mode = st.radio("Pilih sumber gambar", ["Upload dari Galeri", "Ambil dari Kamera"], horizontal=True)

    file = None
    if mode == "Upload dari Galeri":
        file = st.file_uploader("Upload gambar daun", type=["jpg", "jpeg", "png"])
    else:
        file = st.camera_input("Ambil foto daun")

    if file is not None:
        st.image(file, caption="Preview", use_container_width=True)
        if st.button("Mulai Analisis", use_container_width=True):
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            st.session_state.image = img
            st.session_state.page  = "processing"
            st.rerun()

# ============================================================
# HALAMAN 3: PROCESSING
# ============================================================
elif st.session_state.page == "processing":

    bg_base64 = get_base64_image("bground.jpeg")

    st.markdown(f"""
    <style>
    #MainMenu, footer, header {{ visibility: hidden; }}
    .stApp {{
        background: linear-gradient(rgba(10,40,10,0.65), rgba(10,40,10,0.65)),
                    url(data:image/jpeg;base64,{bg_base64});
        background-size: cover; background-position: center; background-attachment: fixed;
    }}
    h1 {{ text-align:center; color:white; }}
    .img-result-wrapper {{ display:flex; justify-content:center; margin:0 auto 1.5rem auto; max-width:480px; }}
    .img-result-wrapper img {{ width:100%; border-radius:12px; border:2px solid rgba(255,255,255,0.3); box-shadow:0 8px 32px rgba(0,0,0,0.45); }}
    .section-title {{ color:white; font-size:1.3rem; font-weight:700; margin-bottom:0.8rem; }}
    .card-sehat {{ border-radius:12px; padding:1.2rem 1.4rem; margin-bottom:1rem; background:rgba(0,180,80,0.25); border:1px solid rgba(0,220,100,0.55); }}
    .card-sakit {{ border-radius:12px; padding:1.2rem 1.4rem; margin-bottom:1rem; background:rgba(220,40,40,0.25); border:1px solid rgba(255,80,80,0.55); }}
    .card-prob  {{ border-radius:12px; padding:1.2rem 1.4rem; margin-bottom:1rem; background:rgba(50,130,220,0.20); border:1px solid rgba(80,160,255,0.45); }}
    .result-badge {{ font-size:1.25rem; font-weight:800; color:#ffffff; letter-spacing:0.04em; margin:0 0 0.8rem 0; }}
    .conf-label {{ color:rgba(255,255,255,0.80); font-size:0.88rem; margin-bottom:5px; }}
    .conf-bg {{ background:rgba(255,255,255,0.15); border-radius:999px; height:10px; overflow:hidden; margin-bottom:1rem; }}
    .conf-fill-sehat {{ height:100%; border-radius:999px; background:linear-gradient(90deg,#5cb85c,#a5d6a7); }}
    .conf-fill-sakit {{ height:100%; border-radius:999px; background:linear-gradient(90deg,#e53935,#ef9a9a); }}
    .ciri-label {{ color:rgba(255,255,255,0.75); font-size:0.85rem; margin-bottom:0.4rem; }}
    .ciri-list {{ color:rgba(255,255,255,0.88); font-size:0.9rem; padding-left:1.2rem; margin:0; line-height:1.8; }}
    .ciri-list li {{ margin-bottom:0.2rem; }}
    .prob-table {{ width:100%; border-collapse:collapse; font-size:0.88rem; }}
    .prob-table th {{ color:rgba(255,255,255,0.55); font-weight:600; text-align:left; padding:4px 8px; border-bottom:1px solid rgba(255,255,255,0.15); }}
    .prob-table td {{ color:rgba(255,255,255,0.85); padding:6px 8px; }}
    .prob-table tr:hover td {{ background:rgba(255,255,255,0.06); }}
    .stButton > button {{ background:rgba(255,255,255,0.12) !important; color:white !important; border:1px solid rgba(255,255,255,0.40) !important; border-radius:8px !important; font-weight:600 !important; }}
    .stButton > button:hover {{ background:rgba(255,255,255,0.25) !important; }}
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1>Hasil Analisis</h1>", unsafe_allow_html=True)

    img = st.session_state.image

    # ── Inferensi TFLite (bukan model.predict) ─────────────────
    with st.spinner("Menganalisis gambar daun..."):
        # 1. Resize dan normalisasi gambar
        img_resized = cv2.resize(img, IMG_SIZE)
        img_array   = img_resized.astype("float32") / 255.0
        img_array   = np.expand_dims(img_array, axis=0)  # shape: (1, 224, 224, 3)

        # 2. Ambil detail input & output tensor dari interpreter
        input_details  = model.get_input_details()
        output_details = model.get_output_details()

        # 3. Masukkan data gambar ke tensor input
        model.set_tensor(input_details[0]['index'], img_array)

        # 4. Jalankan inferensi
        model.invoke()

        # 5. Ambil hasil dari tensor output → shape: (2,)
        preds = model.get_tensor(output_details[0]['index'])[0]

    top1_idx   = int(np.argmax(preds))
    top1_conf  = float(preds[top1_idx])
    top1_label = CLASS_NAMES[top1_idx]
    conf_pct   = round(top1_conf * 100, 1)

    # ── Gambar asli ────────────────────────────────────────────
    _, buf  = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    img_b64 = base64.b64encode(buf).decode()

    st.markdown(f"""
    <div class="img-result-wrapper">
        <img src="data:image/jpeg;base64,{img_b64}" alt="Gambar Dianalisis" />
    </div>
    <p style="text-align:center;color:rgba(255,255,255,0.50);font-size:0.8rem;
              margin-top:-0.8rem;margin-bottom:1.4rem;">Gambar yang Dianalisis</p>
    """, unsafe_allow_html=True)

    # ── Hasil Diagnosis ────────────────────────────────────────
    st.markdown('<p class="section-title">Hasil Diagnosis</p>', unsafe_allow_html=True)

    CIRI_SEHAT = [
        "Warna daun hijau merata dan segar.",
        "Permukaan daun utuh tanpa kerusakan.",
        "Tidak terdapat bercak kuning, cokelat, atau hitam.",
        "Tidak berlubang akibat hama.",
        "Tidak layu atau mengering.",
        "Bentuk daun normal sesuai jenis tanaman.",
    ]
    CIRI_SAKIT = [
        "Daun mengalami layu atau kehilangan kesegaran.",
        "Terdapat bercak kuning, cokelat, atau hitam pada permukaan daun.",
        "Daun berlubang akibat serangan hama.",
        "Tepi daun mengering atau rusak.",
        "Warna daun tidak merata dan mengalami perubahan warna (klorosis).",
        "Daun tampak keriting, cacat, atau mengalami deformasi.",
        "Menunjukkan gejala kerusakan akibat penyakit, hama, atau faktor lingkungan.",
    ]

    if top1_label == "sehat":
        card_css  = "card-sehat"
        badge     = "✅ Daun Sehat"
        fill_css  = "conf-fill-sehat"
        ciri_list = CIRI_SEHAT
    else:
        card_css  = "card-sakit"
        badge     = "⚠️ Daun Sakit"
        fill_css  = "conf-fill-sakit"
        ciri_list = CIRI_SAKIT

    st.markdown(f"""
    <div class="{card_css}">
        <p class="result-badge">{badge}</p>
        <p class="conf-label">Tingkat Keyakinan Model: <strong>{conf_pct}%</strong></p>
        <div class="conf-bg">
            <div class="{fill_css}" style="width:{conf_pct}%;"></div>
        </div>
        <p class="ciri-label">Ciri-ciri daun {"sehat" if top1_label == "sehat" else "sakit"}:</p>
    </div>
    """, unsafe_allow_html=True)

    ciri_html = "".join(f"<li>{c}</li>" for c in ciri_list)
    st.markdown(f"""
    <div class="{card_css}" style="margin-top:-0.8rem;padding-top:0.6rem;">
        <ul class="ciri-list">{ciri_html}</ul>
    </div>
    """, unsafe_allow_html=True)

    # ── Distribusi Probabilitas ────────────────────────────────
    st.markdown('<p class="section-title" style="margin-top:1rem;">Distribusi Probabilitas</p>',
                unsafe_allow_html=True)

    sorted_idx = np.argsort(preds)[::-1]

    rows = ""
    for rank, idx in enumerate(sorted_idx, start=1):
        name     = CLASS_NAMES[int(idx)]
        pct      = round(float(preds[idx]) * 100, 2)
        icon     = "✅" if name == "sehat" else "⚠️"
        is_top   = rank == 1
        td_style = "font-weight:800;color:#ffffff;" if is_top else "color:rgba(255,255,255,0.80);"
        rows += (
            f"<tr>"
            f"<td style='{td_style}'>{rank}</td>"
            f"<td style='{td_style}'>{icon} {name.capitalize()}</td>"
            f"<td style='{td_style}'>{pct}%</td>"
            f"</tr>"
        )

    st.markdown(f"""
    <div class="card-prob">
        <table class="prob-table">
            <thead>
                <tr>
                    <th>#</th><th>Kelas</th><th>Probabilitas</th>
                </tr>
            </thead>
            <tbody>{rows}</tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

    # ── Tombol navigasi ────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        if st.button("← Kembali ke Dashboard", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()
    with col2:
        if st.button("Analisis Gambar Baru", use_container_width=True):
            st.session_state.page = "dashboard"
            st.rerun()