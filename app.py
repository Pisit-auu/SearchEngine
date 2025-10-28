import streamlit as st
import os
import io
import numpy as np
from pathlib import Path
from PIL import Image, ImageFile, ImageOps
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import faiss
import clip

BASE_PATH = Path(os.getenv("DATA_DIR", "data/train")).resolve()
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
EMB_FILE = "clip_embeddings.npy"
FN_FILE  = "clip_filenames.npy"   
INDEX_FILE = "clip_image_index.faiss"

device = "cuda" if torch.cuda.is_available() else "cpu"

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
        st.error(f"ไม่สามารถเปิดไฟล์รูปได้: {e}")
        return None

@st.cache_resource
def load_model():
    st.write("กำลังโหลดโมเดล CLIP")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    st.write("โหลดโมเดล CLIP สำเร็จ")
    return model, preprocess

model, preprocess = load_model()

def get_embedding(image: Image.Image) -> np.ndarray:
    if image is None:
        return None
    x = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model.encode_image(x)
    return feat.squeeze(0).cpu().numpy().astype("float32")


@st.cache_data(show_spinner=True)
def create_or_load_index(base_path_str: str):
    base_dir = Path(base_path_str).resolve()

    if Path(INDEX_FILE).exists() and Path(EMB_FILE).exists() and Path(FN_FILE).exists():
        try:
            st.write("กำลังโหลด embedding เดิม")
            index = faiss.read_index(INDEX_FILE)
            embeddings = np.load(EMB_FILE)
            rel_filenames = np.load(FN_FILE, allow_pickle=True)

            filenames = [str(to_full_path(rel, base_dir)) for rel in rel_filenames]
            st.success(f"โหลดสำเร็จ (มี {index.ntotal} รูป, ขนาด {index.d},)")
            return index, embeddings, np.array(filenames)
        except Exception as e:
            st.warning(f"โหลดไม่สำเร็จ: {e}")

    if not base_dir.exists():
        st.error(f"ไม่พบโฟลเดอร์รูป: {base_dir}\n")
        return None, None, None

    st.write(f"กำลังสร้าง index ใหม่จาก: '{base_dir}'")
    all_files = []
    for root, _, files in os.walk(base_dir):
        for fname in files:
            if fname.lower().endswith(VALID_EXT):
                all_files.append(str(Path(root) / fname))

    if not all_files:
        st.error(f"ไม่พบไฟล์รูปภาพใน: {base_dir}")
        return None, None, None

    embeddings_list, rel_fns = [], []
    progress = st.progress(0, text="กำลังประมวลผลรูปภาพด้วย CLIP")
    total = len(all_files)

    for i, p in enumerate(all_files):
        try:
            img = open_image(p)
            if img:
                emb = get_embedding(img)
                embeddings_list.append(emb)
                rel_fns.append(to_posix_rel(Path(p), base_dir))
        except Exception as e:
            st.warning(f"ข้ามไฟล์เสีย: {p} ({e})")

        progress.progress((i + 1) / total, text=f"ประมวลผล {i+1}/{total}")

    progress.empty()

    if not embeddings_list:
        st.error("ไม่สามารถสร้าง embedding ได้")
        return None, None, None

    embeddings = np.vstack(embeddings_list).astype("float32")
    faiss.normalize_L2(embeddings)  # ใช้ cosine 

    st.write(f"ประมวลผลเสร็จ: {embeddings.shape[0]} รูป, ขนาด {embeddings.shape[1]}-d")
    st.write("กำลังสร้าง FAISS")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    np.save(EMB_FILE, embeddings)
    np.save(FN_FILE, np.array(rel_fns, dtype=object))
    faiss.write_index(index, INDEX_FILE)

    # คืนค่า filenames เป็น full path (เพื่อแสดงภาพ)
    filenames_full = [str(to_full_path(rel, base_dir)) for rel in rel_fns]
    st.success(f"สร้างและบันทึก index สำเร็จ (ทั้งหมด {index.ntotal} รูป)")
    return index, embeddings, np.array(filenames_full)

st.title("Image Search Engine (CLIP + FAISS)")
st.caption("อัปโหลดรูปเพื่อค้นหารูปที่ 'สถานที่เดียวกัน' จากคลังภาพ")

index, embeddings, filenames = create_or_load_index(str(BASE_PATH))
if index is None:
    st.stop()

uploaded_file = st.file_uploader(
    "เลือกรูปภาพที่ต้องการค้นหา",
    type=["jpg", "jpeg", "png", "bmp", "webp"]
)

threshold = st.slider("ตั้งค่าเกณฑ์ที่สามารถรับได้",
                      min_value=0.10, max_value=1.00, value=0.80, step=0.01)

if uploaded_file is not None:
    query_img = open_image(uploaded_file)
    if query_img:
        st.subheader("รูปต้นฉบับ:")
        st.image(query_img, width=240)

        q = get_embedding(query_img).reshape(1, -1)
        faiss.normalize_L2(q)

        D, I = index.search(q, index.ntotal)
        D, I = D[0], I[0]

        results = []
        for idx, sim in zip(I, D):
            if sim < 0.9999 and sim >= threshold:
                results.append((filenames[idx], float(sim * 100.0)))

        results.sort(key=lambda x: x[1], reverse=True)

        st.subheader(f"ผลการค้นหา (≥ {threshold*100:.0f}%): พบ {len(results)} รูป")
        if not results:
            st.warning("ไม่พบรูปที่ผ่านเกณฑ์")
        else:
            ncols = 4
            cols = st.columns(ncols)
            for j, (path, conf) in enumerate(results):
                with cols[j % ncols]:
                    img_res = open_image(path)
                    if img_res:
                        st.image(img_res, caption=f"{conf:.1f}%")
                        st.caption(Path(path).name, help=path)
