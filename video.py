import streamlit as st
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import io
import base64
from PIL import Image
import cv2
import tempfile

# Load environment variables
load_dotenv()

# Configure the Google Generative AI model
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    st.error("Google API Key not found. Please add it to the .env file.")
    st.stop()

# Initialize the model
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-exp-image-generation", google_api_key=api_key)

def extract_frames(video_path, num_frames=5):
    """Extract evenly spaced frames from video"""
    frames = []
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames // num_frames

    for i in range(num_frames):
        video.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = video.read()
        if ret:
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame_rgb))
    
    video.release()
    return frames

def analyze_frame(frame, prompt):
    """Analyze a single frame using Gemini"""
    try:
        # Convert frame to bytes
        img_byte_arr = io.BytesIO()
        frame.save(img_byte_arr, format='JPEG')
        img_bytes_b64 = base64.b64encode(img_byte_arr.getvalue()).decode()

        # Create message
        image_message = {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_bytes_b64}"},
        }
        text_message = {"type": "text", "text": prompt}
        message = HumanMessage(content=[text_message, image_message])
        
        # Get response
        response = llm.invoke([message])
        return response.content
    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")
        return None

# Streamlit app
st.title("Video Content Analyzer")
st.write("Upload a video file, and the AI will analyze its content.")

# File uploader
uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
        tmp_file.write(uploaded_file.read())
        video_path = tmp_file.name

    # Display video
    st.video(video_path)
    
    if st.button("Analyze Video"):
        st.write("Extracting and analyzing frames...")
        
        # Extract frames
        frames = extract_frames(video_path)
        
        # Analyze each frame
        analysis_prompt = "Describe what you see in this frame of the video."
        frame_analyses = []
        
        progress_bar = st.progress(0)
        for i, frame in enumerate(frames):
            analysis = analyze_frame(frame, analysis_prompt)
            if analysis:
                frame_analyses.append(analysis)
            progress_bar.progress((i + 1) / len(frames))
        
        # Combine analyses for overall description
        if frame_analyses:
            st.subheader("Video Analysis Results:")
            combined_prompt = "Based on these frame descriptions, provide a coherent summary of the video content: " + " ".join(frame_analyses)
            
            try:
                final_message = HumanMessage(content=combined_prompt)
                final_analysis = llm.invoke([final_message])
                st.write(final_analysis.content)
            except Exception as e:
                st.error(f"Error generating final analysis: {e}")
        else:
            st.write("Could not analyze the video frames.")
    
    # Cleanup
    os.unlink(video_path)

# Add instructions
st.sidebar.header("How to Run")
st.sidebar.info(
    "1. Create a `.env` file in the same directory as `video.py`.\n"
    "2. Add your Google AI Studio API key to the `.env` file: `GOOGLE_API_KEY='YOUR_API_KEY'`\n"
    "3. Open a terminal in VS Code.\n"
    "4. Activate the virtual environment: `.venv\\Scripts\\activate` (Windows) or `source .venv/bin/activate` (macOS/Linux)\n"
    "5. Run the app: `streamlit run video.py`"
)