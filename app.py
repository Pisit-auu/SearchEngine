import streamlit as st
import os
import io
from collections import Counter
import numpy as np
from pathlib import Path
from PIL import Image, ImageFile, ImageOps
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import faiss
import clip

st.set_page_config(
    page_title="Image Search Engine",
    page_icon=":material/travel_explore:",
    layout="wide",
)

BASE_PATH = Path(os.getenv("DATA_DIR", "data/train")).resolve()
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
EMB_FILE = "clip_embeddings.npy"
FN_FILE  = "clip_filenames.npy"
INDEX_FILE = "clip_image_index.faiss"

device = "cuda" if torch.cuda.is_available() else "cpu"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=Hanken+Grotesk:wght@400;500;600&display=swap');

:root {
    --ink: #1f1b16;
    --muted: #6b6257;
    --accent: #b4532a;
    --line: #e4dccf;
}
html, body, [class*="css"], .stMarkdown, .stCaption, button, input, label {
    font-family: 'Hanken Grotesk', system-ui, sans-serif;
}
h1, h2, h3 {
    font-family: 'Newsreader', Georgia, serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em;
    color: var(--ink);
    text-wrap: balance;
}
h1 { font-size: clamp(2rem, 4vw, 3.1rem) !important; line-height: 1.05 !important; }
.block-container { padding-top: 2.5rem; max-width: 1200px; }
::selection { background: #f0cdb8; color: var(--ink); }
[data-testid="stImage"] img { border-radius: 6px; }
[data-testid="stMetricValue"] { font-variant-numeric: tabular-nums; font-family: 'Newsreader', Georgia, serif; }
.result-meta { display: flex; justify-content: space-between; gap: .5rem; margin: .35rem 0 1.25rem; font-size: .85rem; }
.result-meta .place { color: var(--ink); font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.result-meta .score { color: var(--accent); font-weight: 600; font-variant-numeric: tabular-nums; }
[data-testid="stCaptionContainer"], .stCaption { color: var(--muted) !important; }
.lede { color: var(--muted); font-size: 1.05rem; max-width: 60ch; margin-top: -.5rem; }
</style>
""", unsafe_allow_html=True)

def to_posix_rel(full_path: Path, base_dir: Path) -> str:
    return full_path.resolve().relative_to(base_dir.resolve()).as_posix()

def to_full_path(rel_posix: str, base_dir: Path) -> Path:
    return (base_dir / Path(rel_posix)).resolve()

def open_image(image_source) -> Image.Image:
    try:
        if hasattr(image_source, "read"):
            data = image_source.read()
            im = Image.open(io.BytesIO(data))
        elif isinstance(image_source, (bytes, bytearray, io.BytesIO)):
            im = Image.open(io.BytesIO(image_source if isinstance(image_source, (bytes, bytearray)) else image_source.getvalue()))
        else:
            im = Image.open(image_source)

        im = ImageOps.exif_transpose(im)
        return im.convert("RGB")
    except Exception as e:
        st.error(f"Couldn't open this file: {e}. Try a JPG, PNG, BMP or WEBP image.")
        return None

@st.cache_resource(show_spinner="Loading the CLIP model…")
def load_model():
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    return model, preprocess

model, preprocess = load_model()

def get_embedding(image: Image.Image) -> np.ndarray:
    if image is None:
        return None
    x = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(x)
    return feat.squeeze(0).cpu().numpy().astype("float32")


@st.cache_resource(show_spinner="Loading the image index…")
def create_or_load_index(base_path_str: str):
    base_dir = Path(base_path_str).resolve()

    if Path(INDEX_FILE).exists() and Path(EMB_FILE).exists() and Path(FN_FILE).exists():
        try:
            index = faiss.read_index(INDEX_FILE)
            embeddings = np.load(EMB_FILE)
            rel_filenames = np.load(FN_FILE, allow_pickle=True)

            filenames = [str(to_full_path(rel, base_dir)) for rel in rel_filenames]
            return index, embeddings, np.array(filenames)
        except Exception as e:
            st.warning(f"Saved index couldn't be loaded ({e}). Building a new one.")

    if not base_dir.exists():
        st.error(f"Image folder not found: {base_dir}. Set DATA_DIR or restore data/train.")
        return None, None, None

    all_files = []
    for root, _, files in os.walk(base_dir):
        for fname in files:
            if fname.lower().endswith(VALID_EXT):
                all_files.append(str(Path(root) / fname))

    if not all_files:
        st.error(f"No images found in {base_dir}.")
        return None, None, None

    embeddings_list, rel_fns = [], []
    progress = st.progress(0, text="Embedding images with CLIP")
    total = len(all_files)

    for i, p in enumerate(all_files):
        try:
            img = open_image(p)
            if img:
                emb = get_embedding(img)
                embeddings_list.append(emb)
                rel_fns.append(to_posix_rel(Path(p), base_dir))
        except Exception as e:
            st.warning(f"Skipped unreadable file: {p} ({e})")

        progress.progress((i + 1) / total, text=f"Embedding images {i+1}/{total}")

    progress.empty()

    if not embeddings_list:
        st.error("No embeddings could be created from the images.")
        return None, None, None

    embeddings = np.vstack(embeddings_list).astype("float32")
    faiss.normalize_L2(embeddings)  # ใช้ cosine

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    np.save(EMB_FILE, embeddings)
    np.save(FN_FILE, np.array(rel_fns, dtype=object))
    faiss.write_index(index, INDEX_FILE)

    # คืนค่า filenames เป็น full path (เพื่อแสดงภาพ)
    filenames_full = [str(to_full_path(rel, base_dir)) for rel in rel_fns]
    return index, embeddings, np.array(filenames_full)


def thumb(image: Image.Image) -> Image.Image:
    # ครอปเป็น 4:3 ให้กริดเรียงเสมอกัน
    return ImageOps.fit(image, (480, 360))


def place_of(path: str) -> str:
    # ชื่อโฟลเดอร์แม่ = ชื่อสถานที่
    return Path(path).parent.name


index, embeddings, filenames = create_or_load_index(str(BASE_PATH))
if index is None:
    st.stop()

LOCATIONS = [
    "Antarctica",
    "Burj Khalifa - UAE",
    "Chich-n Itz - Mexico",
    "Christ the Redeemer Statue",
    "Eiffel Tower - Paris",
    "Giant-s Causeway",
    "Great Wall Of China - China",
    "Himalaya - India",
    "Machu Pichu",
    "Niagara Falls",
    "Pyramids Of Giza - Egypt",
    "Roman Colosseum - Rome",
    "Santorini",
    "Statue Of Liberty - NYC",
    "Stonehenge",
    "Taj Mahal - India",
    "The Blue Grotto - Capri",
    "Venezuela Angel Falls",
]

image_paths = [
    "./data/test/2.AtractivoGrande_2352019081130.jpg",
    "./data/test/5.jpg",
    "./data/test/9.jpg",
    "./data/test/37.831300.jpg",
    "./data/test/43.great-wall-of-china-facts-2.jpg",
    "./data/test/48.american-falls1-2__medium.jpg",
    "./data/test/50.machu-picchu-cusco.jpg",
    "./data/test/60.01-eiffel-tower.jpg",
    "./data/test/63.hangchendzonga-national-par.jpg",
    "./data/test/77.jpg",
    "./data/test/79.jpg",
    "./data/test/90.jpg",
    "./data/test/92.jpg",
    "./data/test/110.jpg",
    "./data/test/119.jpg",
    "./data/test/171.jpg",
    "./data/test/180.jpg",
    "./data/test/325.jpg",
]

with st.sidebar:
    st.subheader("Search settings")
    threshold = st.slider("Minimum similarity", min_value=0.10, max_value=1.00,
                          value=0.80, step=0.01, format="%.2f",
                          help="Only pictures at least this similar to yours are shown.")
    max_results = st.select_slider("Show at most", options=[12, 24, 48, 96], value=24,
                                   help="Best matches come first.")
    st.divider()
    st.caption(f"{len(LOCATIONS)} locations · {index.ntotal:,} indexed pictures")
    st.markdown("\n".join(f"- {loc}" for loc in LOCATIONS))

st.title("Image Search Engine")
st.markdown('<p class="lede">Upload a photo of a landmark and find pictures of the same place, '
            'matched with CLIP embeddings and a FAISS index.</p>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose a picture to search with",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    st.session_state.pop("example", None)
    query_img = open_image(uploaded_file)
elif st.session_state.get("example"):
    query_img = open_image(st.session_state["example"])
else:
    query_img = None

if query_img is None:
    st.subheader("No photo handy? Try an example")
    cols = st.columns(6)
    for idx, path in enumerate(image_paths):
        with cols[idx % 6]:
            if os.path.exists(path):
                ex_img = open_image(path)
                if ex_img:
                    st.image(thumb(ex_img), use_column_width=True)
                if st.button("Search", key=f"ex-{idx}", use_container_width=True):
                    st.session_state["example"] = path
                    st.rerun()
            else:
                st.caption(f"Missing: {os.path.basename(path)}")
    st.stop()

q = get_embedding(query_img).reshape(1, -1)
faiss.normalize_L2(q)

D, I = index.search(q, index.ntotal)
D, I = D[0], I[0]

results = []
for idx, sim in zip(I, D):
    if idx != -1 and sim < 0.9999 and sim >= threshold:
        results.append((filenames[idx], float(sim * 100.0)))

results.sort(key=lambda x: x[1], reverse=True)
shown = results[:max_results]

left, right = st.columns([1, 2], gap="large")
with left:
    st.image(query_img, caption="Your picture", use_column_width=True)
    if st.session_state.get("example") and st.button("Clear example"):
        st.session_state.pop("example", None)
        st.rerun()
with right:
    if results:
        top_places = Counter(place_of(p) for p, _ in results[:10])
        best_place = top_places.most_common(1)[0][0]
        st.subheader(best_place)
        st.caption("Most common place among the top matches")
        m1, m2, m3 = st.columns(3)
        m1.metric("Matches", f"{len(results):,}")
        m2.metric("Best match", f"{results[0][1]:.1f}%")
        m3.metric("Threshold", f"{threshold*100:.0f}%")
    else:
        st.subheader("No matches at this threshold")
        st.write(f"Nothing in the index is at least {threshold*100:.0f}% similar. "
                 "Lower **Minimum similarity** in the sidebar to see looser matches.")

if shown:
    st.divider()
    if len(results) > len(shown):
        st.caption(f"Showing the best {len(shown)} of {len(results):,} matches.")
    ncols = 4
    cols = st.columns(ncols)
    for j, (path, conf) in enumerate(shown):
        with cols[j % ncols]:
            img_res = open_image(path)
            if img_res:
                st.image(thumb(img_res), use_column_width=True)
                st.markdown(
                    f'<div class="result-meta" title="{Path(path).name}">'
                    f'<span class="place">{place_of(path)}</span>'
                    f'<span class="score">{conf:.1f}%</span></div>',
                    unsafe_allow_html=True,
                )
