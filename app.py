import os
import base64
import requests
from io import BytesIO

import streamlit as st
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

st.set_page_config(page_title="VibeLens AI", page_icon="🎧", layout="centered")

@st.cache_resource(show_spinner=False)
def load_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

VIBE_LABELS = [
    "happy","joyful","cheerful","playful","fun","excited","energetic",
    "calm","peaceful","serene","relaxing","zen","quiet",
    "moody","melancholy","gloomy","sad","lonely","cold",
    "romantic","love","dreamy","nostalgic","hopeful","uplifting",
    "epic","powerful","dramatic","intense","dark","mysterious"
]

def predict_vibe(image: Image.Image):
    model, processor = load_model()
    inputs = processor(
        text=VIBE_LABELS,
        images=image,
        return_tensors="pt",
        padding=True
    )
    with torch.no_grad():
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image.squeeze()
        probs = logits_per_image.softmax(dim=0)
    top_id = probs.argmax().item()
    vibe = VIBE_LABELS[top_id]
    score = probs[top_id].item()
    return vibe, score

### ---- UI ---- ###
st.title("🎧 VibeLens AI – Image‑to‑Song Recommender")

uploaded = st.file_uploader("Upload an image to find its vibe & get a song", type=["jpg","jpeg","png"])

col_spotify = st.expander("🔑 Spotify API credentials (required for song recommendation)")
with col_spotify:
    SPOTIFY_CLIENT_ID = st.text_input("Client ID", type="password")
    SPOTIFY_CLIENT_SECRET = st.text_input("Client Secret", type="password")

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded image", use_column_width=True)

    with st.spinner("Analyzing vibe…"):
        vibe, confidence = predict_vibe(img)
    st.success(f"Detected vibe: **{vibe}** (confidence: {confidence:.2%})")

    # Only attempt Spotify if creds provided
    if SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET:
        def get_spotify_token(cid, secret):
            auth_str = f"{cid}:{secret}"
            b64_auth = base64.b64encode(auth_str.encode()).decode()
            headers = {"Authorization": f"Basic {b64_auth}"}
            data = {"grant_type":"client_credentials"}
            resp = requests.post("https://accounts.spotify.com/api/token", data=data, headers=headers, timeout=30)
            resp.raise_for_status()
            return resp.json()["access_token"]

        def recommend_song(vibe_word, token):
            headers = {"Authorization": f"Bearer {token}"}
            params = {
                "q": vibe_word,
                "type": "track",
                "limit": 10
            }
            r = requests.get("https://api.spotify.com/v1/search", params=params, headers=headers, timeout=30)
            r.raise_for_status()
            items = r.json()["tracks"]["items"]
            if not items:
                return None
            return items[0]  # the first matching track

        try:
            token = get_spotify_token(SPOTIFY_CLIENT_ID, SPOTIFY_CLIENT_SECRET)
            track = recommend_song(vibe, token)
            if track:
                track_name = track["name"]
                artists = ", ".join([a["name"] for a in track["artists"]])
                track_url = track["external_urls"]["spotify"]
                st.subheader("🎵 Suggested Song")
                st.write(f"[{track_name} – {artists}]({track_url})")
            else:
                st.info("No song found matching that vibe.")
        except Exception as e:
            st.error(f"Spotify request failed: {e}")
    else:
        st.info("Provide Spotify credentials above to get a song recommendation.")
