from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
import tempfile
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import shutil
from video import extract_frames, analyze_frame

# Initialize FastAPI app
app = FastAPI(
    title="Video Analysis API",
    description="API for analyzing video content using Google's Gemini AI"
)

# Load environment variables
load_dotenv()

# Configure the Google Generative AI model
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("Google API Key not found. Please add it to the .env file.")

# Initialize the model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash-exp-image-generation", 
    google_api_key=api_key
)

@app.post("/analyze-video/")
async def analyze_video(video: UploadFile = File(...)):
    """
    Endpoint to analyze video content
    """
    try:
        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            # Copy uploaded file to temporary file
            shutil.copyfileobj(video.file, tmp_file)
            video_path = tmp_file.name

        # Extract frames
        frames = extract_frames(video_path)
        
        # Analyze each frame
        analysis_prompt = "Describe what you see in this frame of the video."
        frame_analyses = []
        
        for frame in frames:
            analysis = analyze_frame(frame, analysis_prompt)
            if analysis:
                frame_analyses.append(analysis)

        # Generate final analysis
        if frame_analyses:
            combined_prompt = "Based on these frame descriptions, provide a coherent summary of the video content: " + " ".join(frame_analyses)
            final_message = HumanMessage(content=combined_prompt)
            final_analysis = llm.invoke([final_message])
            
            response_data = {
                "status": "success",
                "frame_analyses": frame_analyses,
                "summary": final_analysis.content
            }
        else:
            response_data = {
                "status": "error",
                "message": "Could not analyze video frames"
            }

        # Cleanup
        os.unlink(video_path)
        
        return JSONResponse(content=response_data)

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "message": f"An error occurred: {str(e)}"
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)