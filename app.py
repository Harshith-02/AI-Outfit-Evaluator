import streamlit as st
from PIL import Image
from rembg import remove
import torch, clip, io, cv2, numpy as np
from sklearn.cluster import KMeans
import colorsys
import cohere
import os

# -------- Cohere setup --------
cohere_api_key = os.getenv("COHERE_API_KEY")
co = cohere.Client(cohere_api_key) if cohere_api_key else None

# -------- Page config --------
st.set_page_config(page_title="AI Outfit Evaluator", page_icon="🤵", layout="centered")
st.title("🤵 AI Outfit Evaluator")
st.caption("Upload an outfit photo → background removed, style & colour harmony scored.")

# -------- Sidebar options --------
st.sidebar.header("Options")
feedback_type = st.sidebar.radio("Feedback detail level", ["Short Verdict", "Detailed Analysis"])
target_occasion = st.sidebar.selectbox(
    "Target Occasion",
    ["Casual Outing", "Interview", "Wedding", "Date", "Party", "Business Meeting", "None"]
)
compare_mode = st.sidebar.checkbox("Compare with another outfit")

# -------- Upload first outfit --------
uploaded_file = st.file_uploader("Upload an outfit image", type=["jpg", "jpeg", "png"], key="first")
if uploaded_file is None:
    st.warning("Please upload the first outfit image to proceed.")
    st.stop()

# -------- Upload second outfit if compare_mode --------
uploaded_file2 = None
if compare_mode:
    uploaded_file2 = st.file_uploader("Upload second outfit image for comparison", type=["jpg", "jpeg", "png"], key="second")
    if uploaded_file2 is None:
        st.info("Please upload the second outfit image to compare.")
        st.stop()

# -------- Helper: Background removal --------
def remove_bg(img):
    with st.spinner("Removing background…"):
        result = remove(img)
        bg_removed_img = (
            Image.open(io.BytesIO(result)).convert("RGBA")
            if isinstance(result, bytes) else result.convert("RGBA")
        )
    return bg_removed_img

# -------- Helper: CLIP model loading --------
@st.cache_resource(show_spinner=False)
def load_clip():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model, preprocess = clip.load("ViT-B/32", device=device)
    except Exception as e:
        st.error(f"Failed to load CLIP model: {e}")
        st.stop()
    return model, preprocess, device

model, preprocess, device = load_clip()

# -------- Analyze outfit --------
def analyze_outfit(image, label="Outfit"):
    st.markdown(f"### {label}")
    st.image(np.array(image))

    bg_removed = remove_bg(image)
    st.image(np.array(bg_removed), caption="Background Removed")

    # Color harmony
    np_img = cv2.cvtColor(np.array(bg_removed.convert("RGB")), cv2.COLOR_RGB2BGR)
    pixels = np_img.reshape(-1, 3)
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10).fit(pixels)
    centers = kmeans.cluster_centers_.astype("uint8")

    hues = [colorsys.rgb_to_hsv(*(c / 255))[0] * 360 for c in centers]
    def hue_gap(a, b): d = abs(a - b) % 360; return min(d, 360 - d)
    gaps = [hue_gap(h1, h2) for i, h1 in enumerate(hues) for h2 in hues[i + 1:]]
    mean_gap = float(np.mean(gaps))
    harmony_raw = min(abs(mean_gap - 30), abs(mean_gap - 180))
    harmony = max(0, 1 - harmony_raw / 180)

    st.subheader("🎨 Colour Harmony")
    st.metric("Harmony Score", f"{harmony * 100:.1f} / 100")

    cols = st.columns(3)
    for i, rgb in enumerate(centers):
        swatch = Image.new("RGB", (80, 80), tuple(map(int, rgb)))
        cols[i].image(swatch, caption=f"Colour {i + 1}")

    # Style classification using CLIP
    labels = [
        "casual", "formal", "sporty", "ethnic",
        "party", "business", "vintage", "streetwear"
    ]
    img_tensor = preprocess(bg_removed).unsqueeze(0).to(device)
    text_tokens = torch.cat([clip.tokenize(f"This is a {l} outfit") for l in labels]).to(device)

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
            st.write(f"{l.capitalize()} – {s * 100:.2f}%")

    return {
        "style_score": float(similarity[top_idx]),
        "color_score": harmony,
        "predicted_style": predicted_style,
        "colors": [tuple(int(v) for v in rgb) for rgb in centers],
    }

# -------- Analyze first outfit --------
image1 = Image.open(uploaded_file).convert("RGB")
outfit1_data = analyze_outfit(image1, label="First Outfit")

# -------- Analyze second outfit if compare --------
outfit2_data = None
if compare_mode and uploaded_file2 is not None:
    image2 = Image.open(uploaded_file2).convert("RGB")
    outfit2_data = analyze_outfit(image2, label="Second Outfit")

# -------- Compose prompt for AI feedback --------
def make_prompt(data):
    prompt_base = f"""
Evaluate this outfit:
Style: {data['predicted_style']}
Colours (RGB): {data['colors']}
Weather score: 80 / 100  # Placeholder

Target occasion: {target_occasion if target_occasion != "None" else "No specific occasion"}

Please provide a {'brief verdict with main points' if feedback_type == 'Short Verdict' else 'detailed analysis including suggestions'}.
"""
    return prompt_base.strip()

# -------- Function to get AI feedback --------
def get_ai_feedback(prompt):
    if not co:
        st.info("Set the COHERE_API_KEY environment variable to enable AI feedback.")
        return None
    with st.spinner("Getting AI feedback..."):
        try:
            resp = co.chat(
                model="command-nightly",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.7,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            st.error(f"Cohere error: {e}")
            return None

# -------- Display feedback for first outfit --------
prompt1 = make_prompt(outfit1_data)
feedback1 = get_ai_feedback(prompt1)

if feedback1:
    st.subheader("⭐ AI Feedback for First Outfit")
    if feedback_type == "Short Verdict":
        st.success(feedback1)
    else:
        st.write(feedback1)

# -------- Compare outfits --------
if compare_mode and outfit2_data:
    prompt2 = make_prompt(outfit2_data)
    feedback2 = get_ai_feedback(prompt2)

    if feedback2:
        st.subheader("⭐ AI Feedback for Second Outfit")
        if feedback_type == "Short Verdict":
            st.success(feedback2)
        else:
            st.write(feedback2)

    # Simple comparison logic
    st.subheader("⚖️ Outfit Comparison")
    score1 = outfit1_data["style_score"] * 0.4 + outfit1_data["color_score"] * 0.3 + 0.3 * 0.8
    score2 = outfit2_data["style_score"] * 0.4 + outfit2_data["color_score"] * 0.3 + 0.3 * 0.8
    if score1 > score2:
        st.write("🤵 **First outfit is a better choice for your target occasion!**")
    elif score2 > score1:
        st.write("🤵 **Second outfit is a better choice for your target occasion!**")
    else:
        st.write("🤵 **Both outfits are equally good!**")

# -------- Save feedback history --------
def save_feedback(filename, feedback):
    with open(filename, "a", encoding="utf-8") as f:
        f.write("\n\n---\n")
        f.write(f"Feedback:\n{feedback}\n")

if st.button("💾 Save Feedback for First Outfit"):
    if feedback1:
        save_feedback("feedback_history.txt", feedback1)
        st.success("Feedback saved to feedback_history.txt")

if compare_mode and st.button("💾 Save Feedback for Second Outfit"):
    if feedback2:
        save_feedback("feedback_history.txt", feedback2)
        st.success("Feedback saved to feedback_history.txt")

# -------- Fashion Recommendations --------
st.subheader("🛍️ Fashion Recommendations")
rec_styles = {
    "Casual": ["White Sneakers", "Blue Jeans", "Graphic Tee"],
    "Formal": ["Black Blazer", "White Shirt", "Leather Shoes"],
    "Sporty": ["Running Shoes", "Athletic Shorts", "Moisture-Wicking Shirt"],
    "Ethnic": ["Kurta", "Ethnic Jewelry", "Juttis"],
    "Party": ["Sequin Dress", "Heels", "Clutch Bag"],
    "Business": ["Suit", "Tie", "Oxford Shoes"],
    "Vintage": ["High-Waisted Pants", "Polka Dot Blouse", "Loafers"],
    "Streetwear": ["Hoodie", "Sneakers", "Cap"]
}
recommended_items = rec_styles.get(outfit1_data["predicted_style"], ["Classic T-Shirt", "Jeans", "Sneakers"])
st.write(f"Recommended items for your **{outfit1_data['predicted_style']}** style:")
st.write(", ".join(recommended_items))
