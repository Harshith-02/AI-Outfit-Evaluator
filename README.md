# 👗 AI Outfit Evaluator

A Streamlit app that analyzes and evaluates outfits using AI. It removes the image background, classifies the outfit style using CLIP, scores colour harmony with K-Means clustering, and provides feedback using the Cohere AI API.

---

## 🖼️ Features

- ✅ Background removal using `rembg`
- 🎯 Style prediction using OpenAI’s CLIP model
- 🎨 Colour harmony score using K-Means and HSV color space
- 🧠 Smart fashion feedback using Cohere's AI
- 📥 Downloadable transparent-background image

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Outfit-Evaluator.git
cd AI-Outfit-Evaluator
```

### 2. Set up virtual environment (optional but recommended)

```bash
python -m venv .venv
.venv\Scripts\activate        # For Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your Cohere API key

You can either set the key as an environment variable:

```powershell
$env:COHERE_API_KEY="your-cohere-api-key-here"
```

Or put it in a `.env` file (optional, if you use `python-dotenv`):

```env
COHERE_API_KEY=your-cohere-api-key-here
```

---

## ▶️ Run the App

```bash
streamlit run app.py
```

Then open the local URL in your browser.

---

## 📦 `requirements.txt`

```text
streamlit
rembg
torch
opencv-python
Pillow
numpy
scikit-learn
cohere
ftfy
regex
tqdm
git+https://github.com/openai/CLIP.git
```

---

## 📝 Output Example

- ✅ Uploaded image shown
- ✅ Transparent background preview
- ✅ Predicted style: e.g., "Streetwear"
- ✅ Colour harmony score: 78.5 / 100
- ✅ Final rating: 85.3 / 100
- ✅ AI feedback: "This outfit works well for casual events, but could benefit from warmer colors..."

---

## 🌐 Live Demo (Optional)

Coming soon on Streamlit Cloud...

---

## 👤 Author

**Harshith Ankarapu**  
GitHub: [@Harshith-02](https://github.com/Harshith-02)  
Email: harshithankarapu92@gmail.com

---

## 🛡️ License

This project is licensed under the MIT License.
