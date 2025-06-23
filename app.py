# ----------------- Imports -----------------
import streamlit as st
from PIL import Image                      # for colour swatches
from rembg import remove
import torch, clip, io, cv2, numpy as np
from sklearn.cluster import KMeans
import colorsys
import cohere                              # Cohere SDK
import os

# ----------------- Cohere key -----------------
cohere_api_key = os.getenv("COHERE_API_KEY")
co = cohere.Client(cohere_api_key) if cohere_api_key else None

# ----------------- Streamlit page -----------------
st.set_page_config(page_title="AI Outfit Evaluator",
                   page_icon="👗",
                   layout="centered")
st.title("👗 AI Outfit Evaluator")
st.caption("Upload an outfit photo → background removed, style & colour harmony scored.")

# ----------------- File uploader -----------------
uploaded_file = st.file_uploader("Upload an outfit image",
                                 type=["jpg", "jpeg", "png"])

if not uploaded_file:
    st.stop()  # wait until user uploads an image

# ----------------- 1. Original image -----------------
input_image = Image.open(uploaded_file).convert("RGB")
st.image(input_image, caption="Uploaded Image", use_container_width=True)

# ----------------- 2. Remove background -----------------
with st.spinner("Removing background…"):
    result = remove(input_image)
    bg_removed = (
        Image.open(io.BytesIO(result)).convert("RGBA")
        if isinstance(result, bytes)
        else result.convert("RGBA")
    )
st.image(bg_removed, caption="Background Removed",
         use_container_width=True)

# Offer download
buf = io.BytesIO()
bg_removed.save(buf, format="PNG")
st.download_button("📥 Download BG-Removed Image",
                   data=buf.getvalue(),
                   file_name="outfit_no_bg.png",
                   mime="image/png")

# ----------------- 3. Load CLIP -----------------
@st.cache_resource(show_spinner=False)
def load_clip():
    dv = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=dv)
    return model, preprocess, dv

model, preprocess, device = load_clip()

# ----------------- 4. Style classification -----------------
with st.spinner("Analyzing style…"):
    labels = [
        "casual", "formal", "sporty", "ethnic",
        "party", "business", "vintage", "streetwear"
    ]
    img_tensor = preprocess(bg_removed).unsqueeze(0).to(device)
    text_tokens = torch.cat(
        [clip.tokenize(f"This is a {l} outfit") for l in labels]
    ).to(device)

    with torch.no_grad():
        img_feat = model.encode_image(img_tensor)
        txt_feat = model.encode_text(text_tokens)
        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
        similarity = (img_feat @ txt_feat.T).softmax(dim=-1).cpu().numpy()[0]

top_idx = int(similarity.argmax())
predicted_style = labels[top_idx].capitalize()
st.markdown(f"### 🔍 Predicted Style: **{predicted_style}**")

with st.expander("Confidence scores"):
    for s, l in zip(similarity, labels):
        st.progress(float(s), text=f"{l.capitalize()} – {s*100:.2f}%")

# ----------------- 5. Colour harmony -----------------
st.subheader("🎨 Colour Harmony")

np_img  = cv2.cvtColor(np.array(bg_removed.convert("RGB")),
                       cv2.COLOR_RGB2BGR)
pixels  = np_img.reshape(-1, 3)
km      = KMeans(n_clusters=3, random_state=0).fit(pixels)
centers = km.cluster_centers_.astype("uint8")

# Hue harmony
hues = [colorsys.rgb_to_hsv(*(c / 255))[0] * 360 for c in centers]

def hue_gap(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)

gaps        = [hue_gap(h1, h2) for i, h1 in enumerate(hues)
                                  for h2 in hues[i+1:]]
mean_gap    = float(np.mean(gaps))
harmony_raw = min(abs(mean_gap - 30), abs(mean_gap - 180))
harmony     = max(0, 1 - harmony_raw / 180)   # 0–1 scale

st.metric("Harmony Score", f"{harmony*100:.1f} / 100")

swatches = st.columns(3)
for i, rgb in enumerate(centers):
    sw_img = Image.new("RGB", (80, 80), tuple(map(int, rgb)))
    swatches[i].image(sw_img, caption=f"Colour {i+1}",
                      use_container_width=True)

# ----------------- 6. Final rating + Cohere feedback -----------------
st.subheader("⭐ Final Rating & Feedback")

style_score   = float(similarity[top_idx])  # 0–1
color_score   = harmony                     # 0–1
weather_score = 0.8                         # placeholder

final_score = style_score*0.4 + color_score*0.3 + weather_score*0.3
st.metric("Final Score", f"{final_score*100:.1f} / 100")

prompt = f"""
Evaluate this outfit:
Style: {predicted_style}
Colours (RGB): {[tuple(int(v) for v in rgb) for rgb in centers]}
Weather score: {weather_score*100:.1f} / 100

Is the outfit appropriate? Suggest improvements.
""".strip()

if not co:
    st.info("Set the COHERE_API_KEY environment variable to enable AI feedback.")
else:
    with st.spinner("Getting AI feedback…"):
        try:
            resp = co.chat(
                model="command-r",
                message=prompt
            )
            feedback = resp.text.strip()
            st.markdown("### 📝 AI Feedback")
            st.write(feedback)
        except Exception as e:
            st.error(f"Cohere error: {e}")
