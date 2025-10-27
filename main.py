import os
from pathlib import Path

import streamlit as st
from PIL import Image
import google.generativeai as genai

# ----------------------------
# CONFIG
# ----------------------------
# Prefer Streamlit secrets, fallback to env var
API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY"))
if not API_KEY:
    st.error("Missing API key. Add GEMINI_API_KEY to Streamlit secrets or environment.")
    st.stop()

genai.configure(api_key=API_KEY)

SYSTEM_PROMPT = """You are an advanced AI medical image analysis system.
Medical image analysis uses computer algorithms to analyze X-rays, CT, MRI, and ultrasound images to assist diagnosis and treatment.
You will be provided with a medical image. Your task is to analyze the image and provide a detailed report of findings, including any abnormalities or potential health issues.
The report must include:
- A concise description of the image (modality, view, key structures visible)
- Any abnormalities or potential health issues identified (with likelihood/uncertainty)
- Recommendations for further tests or clinical correlation if necessary
Write the report in clear Markdown, for both clinicians and patients. Include numbered sections and bullet points where helpful.
"""

# Model + generation configuration
generation_config = {
    "temperature": 0.7,
    "max_output_tokens": 2048,
    "top_p": 0.95,
    "top_k": 40,
    "response_mime_type": "text/plain"
}
model = genai.GenerativeModel(
    model_name="gemini-2.5-pro",
    system_instruction=SYSTEM_PROMPT,
    generation_config=generation_config
)

# ----------------------------
# LAYOUT
# ----------------------------
try:
    st.set_page_config(
        page_title="Medical Image Analysis AI",
        layout="wide",
        page_icon="images/an icon for a medica.png",  # use forward slashes on Windows too
    )
except Exception:
    # If the icon is missing, continue without failing
    pass

col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    # Show banner if present, else skip silently
    banner_path = Path("images/Mdeical Image AnalysisAI.png")
    if banner_path.exists():
        st.image(str(banner_path), width=200)

st.title("Medical Image Analysis AI")
st.caption("Upload a medical image (X-ray, CT, MRI, ultrasound) for an AI-assisted analysis report.")

uploaded_file = st.file_uploader(
    "Please upload a medical image for analysis",
    type=["png", "jpg", "jpeg", "bmp", "tiff", "tif"],
)

analyze = st.button("Analyze Image")

# ----------------------------
# MAIN LOGIC
# ----------------------------
if analyze:
    if uploaded_file is None:
        st.warning("Please upload an image first.")
        st.stop()

    with st.spinner("Analyzing the medical image..."):
        # Save uploaded file to a temporary location (optional but handy)
        tmp_dir = Path("temp_medical_image")
        tmp_dir.mkdir(parents=True, exist_ok=True)
        temp_image_path = tmp_dir / uploaded_file.name

        with open(temp_image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Open via PIL
        try:
            image = Image.open(temp_image_path).convert("RGB")
        except Exception as e:
            st.error(f"Could not open the uploaded file as an image: {e}")
            st.stop()

        # Show a preview
        st.subheader("Preview")
        st.image(image, caption="Uploaded image", use_container_width=True)

        # Prompt to accompany the image
        user_prompt = (
            "Analyze this medical image. Identify modality/view if possible, "
            "describe notable findings, mention potential abnormalities with uncertainty, "
            "and suggest next steps (tests/clinical correlation) if appropriate."
        )

        try:
            # For multimodal input, pass a list with [image, text]
            response = model.generate_content([image, user_prompt])

            # Some SDK versions require .text, some return candidates; handle both robustly
            report_md = getattr(response, "text", None)
            if not report_md and hasattr(response, "candidates") and response.candidates:
                report_md = response.candidates[0].content.parts[0].text  # fallback

            if not report_md:
                st.error("The model returned an empty response.")
            else:
                st.markdown("### Medical Image Analysis Report")
                st.markdown(report_md)

        except Exception as e:
            st.error(f"Model error: {e}")

        # Optional: clean up (keep file if you want caching)
        try:
            temp_image_path.unlink(missing_ok=True)
        except Exception:
            pass
