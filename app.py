import cv2
import numpy as np
from pydub import AudioSegment
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import shutil

# Initialize Flask app
app = Flask(__name__)



UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load models for stress detection
facial_expression_model = load_model("models/FE_MODEL.h5")
vocal_analysis_model = load_model("models/Vocal_MODEL.h5")

# Function to process face image for facial expression analysis
def process_face_image(image):
    image_resized = cv2.resize(image, (48, 48))  # Adjust based on model input
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_GRAY2RGB)
    image_rgb = image_rgb / 255.0  # Normalize
    image_rgb = np.expand_dims(image_rgb, axis=0)
    return image_rgb

# Function to process spectrogram for vocal analysis
def process_spectrogram(spectrogram):
    spectrogram = np.expand_dims(spectrogram, axis=0)  
    spectrogram = np.expand_dims(spectrogram, axis=-1)  
    spectrogram_resized = tf.image.resize(spectrogram, (128, 128))
    return spectrogram_resized

# Function to extract audio from a video file
def extract_audio_from_video(video_path, output_audio_path):
    # Using pydub to extract audio
    video = AudioSegment.from_file(video_path)
    video.export(output_audio_path, format="wav")
    return output_audio_path

# Function to analyze stress based on facial expressions and vocal analysis
def analyze_stress(video_path):
    cap = cv2.VideoCapture(video_path)
    facial_expression_outputs = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        processed_face = process_face_image(gray_frame)
        facial_expression_output = facial_expression_model.predict(processed_face)
        facial_expression_outputs.append(facial_expression_output)

    cap.release()

    # Extract audio from video
    audio_path = os.path.join(UPLOAD_FOLDER, "audio.wav")
    extract_audio_from_video(video_path, audio_path)

    # Load and process audio with librosa
    y, sr = librosa.load(audio_path, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    processed_spectrogram = process_spectrogram(S_dB)

    vocal_analysis_output = vocal_analysis_model.predict(processed_spectrogram)

    # Weighted stress level calculation
    vocal_accuracy = 0.9
    facial_accuracy = 0.8
    total_accuracy = vocal_accuracy + facial_accuracy
    vocal_weight = vocal_accuracy / total_accuracy
    facial_weight = facial_accuracy / total_accuracy

    vocal_probability = vocal_analysis_output[0][0]
    facial_expression_outputs = np.array(facial_expression_outputs)
    if facial_expression_outputs.ndim == 3:
        facial_expression_outputs = np.squeeze(facial_expression_outputs, axis=1)

    average_probabilities = np.mean(facial_expression_outputs, axis=0)
    facial_probability = average_probabilities[0]

    final_combined_probability = (vocal_weight * vocal_probability) + (facial_weight * facial_probability)
    final_stress_level = "High" if final_combined_probability >= 0.5 else "Low"

    return final_stress_level, round(final_combined_probability, 2)

@app.route("/")
def index():
    return render_template("index.html")

# Sample clustered dataset (replace this with your actual dataset)
clustered_data = {
    "If the probability is below 0.2, listen to these songs.": [
        "disco.00067.wav", "disco.00039.wav", "disco.00078.wav", "disco.00042.wav",
        "disco.00084.wav", "disco.00058.wav", "disco.00095.wav", "disco.00046.wav",
        "disco.00070.wav", "disco.00062.wav", "disco.00093.wav", "disco.00052.wav",
        "disco.00094.wav", "disco.00080.wav", "disco.00048.wav", "disco.00032.wav",
        "disco.00059.wav", "disco.00064.wav", "disco.00050.wav", "disco.00025.wav",
        "disco.00096.wav", "disco.00092.wav", "disco.00081.wav", "disco.00077.wav",
        "disco.00045.wav", "disco.00033.wav", "disco.00066.wav", "disco.00044.wav",
        "disco.00049.wav", "disco.00000.wav", "disco.00079.wav",
        "jazz.00093.wav", "jazz.00092.wav", "jazz.00029.wav", "jazz.00078.wav",
        "pop.00081.wav", "pop.00085.wav", "pop.00069.wav", "pop.00011.wav",
        "blues.00077.wav", "blues.00064.wav", "blues.00079.wav", "blues.00042.wav",
        "hiphop.00082.wav", "hiphop.00016.wav", "hiphop.00025.wav", "hiphop.00046.wav",
        "country.00033.wav", "country.00082.wav", "country.00040.wav", "country.00043.wav",
        "reggae.00070.wav", "reggae.00078.wav", "reggae.00062.wav", "reggae.00060.wav",
        "classical.00052.wav","jazz.00014.wav", "jazz.00036.wav", "jazz.00094.wav", "jazz.00001.wav",
        "jazz.00006.wav", "jazz.00097.wav", "jazz.00039.wav", "jazz.00046.wav",
        "blues.00023.wav", "blues.00086.wav", "blues.00097.wav", "blues.00039.wav",
        "country.00071.wav", "country.00089.wav", "country.00076.wav", "country.00021.wav",
        "reggae.00083.wav", "reggae.00089.wav",
        "classical.00017.wav", "classical.00081.wav", "classical.00088.wav", "classical.00026.wav",
        "classical.00007.wav", "classical.00072.wav", "classical.00013.wav", "classical.00036.wav"
    ],
   
    "If the probability is more than 0.2 and less than 0.4, listen to these songs.": [

        "disco.00031.wav", "disco.00013.wav", "disco.00026.wav", "disco.00030.wav", "disco.00036.wav",
        "disco.00029.wav", "disco.00018.wav", "disco.00038.wav", "disco.00004.wav", "disco.00021.wav",
        "disco.00011.wav", "disco.00023.wav", "jazz.00056.wav", "jazz.00083.wav", "jazz.00081.wav",
        "pop.00037.wav", "pop.00015.wav", "pop.00099.wav", "pop.00034.wav", "pop.00066.wav",
        "pop.00095.wav", "pop.00089.wav", "pop.00036.wav", "pop.00044.wav", "pop.00080.wav",
        "pop.00088.wav", "pop.00084.wav", "pop.00092.wav", "pop.00082.wav", "pop.00078.wav",
        "pop.00093.wav", "pop.00025.wav", "pop.00047.wav", "pop.00028.wav", "pop.00009.wav",
        "pop.00035.wav", "pop.00029.wav", "pop.00068.wav", "pop.00002.wav", "pop.00021.wav",
        "pop.00038.wav", "pop.00022.wav", "pop.00031.wav", "pop.00064.wav", "pop.00096.wav",
        "pop.00030.wav", "pop.00024.wav", "pop.00070.wav", "pop.00050.wav", "hiphop.00053.wav",
        "hiphop.00031.wav", "hiphop.00071.wav", "hiphop.00080.wav", "hiphop.00067.wav",
        "hiphop.00049.wav", "hiphop.00026.wav", "hiphop.00037.wav", "hiphop.00047.wav",
        "hiphop.00069.wav", "country.00009.wav", "reggae.00052.wav", "reggae.00087.wav",
        "reggae.00064.wav","disco.00085.wav", "disco.00071.wav", "disco.00009.wav", "disco.00053.wav", "disco.00007.wav",
        "disco.00043.wav", "disco.00027.wav", "disco.00068.wav", "disco.00003.wav", "disco.00001.wav",
        "disco.00010.wav", "disco.00014.wav", "disco.00097.wav", "jazz.00073.wav", "jazz.00091.wav",
        "jazz.00080.wav", "jazz.00075.wav", "jazz.00088.wav", "jazz.00084.wav", "pop.00083.wav",
        "pop.00094.wav", "pop.00027.wav", "pop.00017.wav", "pop.00046.wav", "pop.00042.wav",
        "pop.00019.wav", "pop.00061.wav", "pop.00004.wav", "pop.00045.wav", "pop.00033.wav",
        "pop.00008.wav", "hiphop.00093.wav", "hiphop.00045.wav", "hiphop.00029.wav", "hiphop.00079.wav",
        "hiphop.00033.wav", "hiphop.00034.wav", "hiphop.00006.wav", "hiphop.00035.wav", "hiphop.00070.wav",
        "hiphop.00042.wav", "hiphop.00075.wav", "hiphop.00074.wav", "hiphop.00073.wav", "hiphop.00039.wav",
        "hiphop.00077.wav", "country.00001.wav", "country.00004.wav", "country.00011.wav",
        "country.00005.wav", "country.00014.wav", "country.00006.wav", "country.00013.wav",
        "country.00038.wav", "country.00007.wav", "country.00000.wav", "reggae.00050.wav",
        "reggae.00072.wav", "reggae.00061.wav", "reggae.00071.wav"



    ],

    "If the probability is more than 0.4 and less than 0.6, listen to these songs.": ["disco.00034.wav", "disco.00035.wav", "disco.00037.wav", "disco.00002.wav",
        "disco.00028.wav", "disco.00022.wav", "jazz.00086.wav", "jazz.00082.wav",
        "pop.00091.wav", "pop.00073.wav", "pop.00016.wav", "pop.00075.wav",
        "pop.00097.wav", "pop.00059.wav", "pop.00067.wav", "pop.00001.wav",
        "pop.00018.wav", "pop.00060.wav", "pop.00000.wav", "pop.00087.wav",
        "pop.00020.wav", "pop.00074.wav", "pop.00086.wav", "pop.00052.wav",
        "pop.00032.wav", "pop.00055.wav", "pop.00039.wav", "pop.00077.wav",
        "pop.00023.wav", "pop.00098.wav", "pop.00048.wav", "pop.00071.wav",
        "pop.00054.wav", "pop.00051.wav", "pop.00076.wav", "pop.00057.wav",
        "pop.00056.wav", "pop.00090.wav", "hiphop.00030.wav", "hiphop.00054.wav",
        "country.00039.wav", "reggae.00086.wav", "reggae.00088.wav", "reggae.00045.wav",
        "reggae.00051.wav","disco.00016.wav", "disco.00024.wav", "disco.00098.wav", "disco.00090.wav",
        "disco.00054.wav", "disco.00005.wav", "disco.00008.wav", "disco.00061.wav",
        "disco.00012.wav", "disco.00072.wav", "disco.00015.wav", "disco.00017.wav",
        "disco.00063.wav", "disco.00074.wav", "disco.00099.wav", "disco.00040.wav",
        "disco.00051.wav", "disco.00075.wav", "disco.00082.wav", "disco.00006.wav",
        "disco.00073.wav", "jazz.00021.wav", "jazz.00085.wav", "jazz.00061.wav",
        "jazz.00050.wav", "jazz.00077.wav", "jazz.00063.wav", "jazz.00079.wav",
        "pop.00053.wav", "pop.00043.wav", "pop.00065.wav", "pop.00062.wav",
        "pop.00072.wav", "pop.00026.wav", "pop.00058.wav", "pop.00012.wav",
        "blues.00074.wav", "blues.00041.wav", "blues.00065.wav", "blues.00063.wav",
        "blues.00072.wav", "blues.00076.wav", "hiphop.00001.wav", "hiphop.00008.wav",
        "hiphop.00064.wav", "hiphop.00041.wav", "hiphop.00089.wav", "hiphop.00051.wav",
        "hiphop.00005.wav", "hiphop.00088.wav", "hiphop.00097.wav", "hiphop.00085.wav",
        "hiphop.00098.wav", "hiphop.00087.wav", "hiphop.00036.wav", "hiphop.00040.wav",
        "hiphop.00028.wav", "hiphop.00018.wav", "hiphop.00092.wav", "hiphop.00091.wav",
        "hiphop.00048.wav", "hiphop.00063.wav", "hiphop.00095.wav", "hiphop.00000.wav",
        "hiphop.00027.wav", "hiphop.00058.wav", "hiphop.00068.wav", "hiphop.00065.wav",
        "hiphop.00020.wav", "hiphop.00052.wav", "country.00034.wav", "country.00045.wav",
        "country.00041.wav", "country.00042.wav", "country.00008.wav", "reggae.00054.wav",
        "reggae.00058.wav", "reggae.00080.wav", "reggae.00090.wav", "reggae.00055.wav",
        "reggae.00048.wav", "reggae.00069.wav", "reggae.00081.wav", "reggae.00095.wav",
        "reggae.00082.wav", "reggae.00044.wav", "reggae.00049.wav", "reggae.00079.wav",
        "reggae.00059.wav"],

    "If the probability is more than 0.6 and less than 0.8, listen to these songs.": [

         "disco.00091.wav", "disco.00057.wav", "disco.00019.wav", "disco.00088.wav",
        "disco.00069.wav", "disco.00076.wav", "disco.00055.wav", "disco.00060.wav",
        "disco.00089.wav", "disco.00083.wav", "disco.00020.wav", "disco.00087.wav",
        "disco.00065.wav", "jazz.00018.wav", "jazz.00023.wav", "jazz.00017.wav",
        "jazz.00028.wav", "jazz.00024.wav", "jazz.00013.wav", "jazz.00087.wav",
        "jazz.00096.wav", "jazz.00095.wav", "jazz.00047.wav", "jazz.00049.wav",
        "jazz.00090.wav", "jazz.00098.wav", "jazz.00069.wav", "pop.00007.wav",
        "pop.00014.wav", "pop.00041.wav", "pop.00010.wav", "pop.00079.wav",
        "pop.00049.wav", "blues.00081.wav", "blues.00053.wav", "blues.00045.wav",
        "blues.00054.wav", "blues.00058.wav", "blues.00059.wav", "blues.00084.wav",
        "blues.00055.wav", "blues.00073.wav", "blues.00050.wav", "blues.00082.wav",
        "hiphop.00022.wav", "hiphop.00007.wav", "hiphop.00032.wav", "hiphop.00086.wav",
        "hiphop.00011.wav", "hiphop.00094.wav", "hiphop.00072.wav", "hiphop.00038.wav",
        "hiphop.00012.wav", "hiphop.00059.wav", "hiphop.00084.wav", "country.00037.wav",
        "country.00048.wav", "country.00015.wav", "country.00084.wav", "country.00003.wav",
        "country.00080.wav", "country.00031.wav", "country.00055.wav", "country.00058.wav",
        "country.00081.wav", "country.00018.wav", "country.00032.wav", "country.00086.wav",
        "country.00070.wav", "country.00099.wav", "reggae.00011.wav", "reggae.00036.wav",
        "reggae.00010.wav", "reggae.00065.wav", "reggae.00003.wav", "reggae.00053.wav",
        "reggae.00098.wav", "reggae.00015.wav", "reggae.00018.wav", "reggae.00021.wav",
        "reggae.00022.wav", "reggae.00091.wav", "reggae.00068.wav", "reggae.00030.wav",
        "reggae.00092.wav", "reggae.00004.wav", "reggae.00097.wav", "reggae.00041.wav",
        "classical.00091.wav", "classical.00090.wav", "classical.00089.wav", "classical.00049.wav","disco.00047.wav", "jazz.00040.wav", "jazz.00038.wav", "jazz.00019.wav",
        "jazz.00072.wav", "jazz.00066.wav", "jazz.00058.wav", "jazz.00041.wav",
        "jazz.00032.wav", "jazz.00008.wav", "jazz.00016.wav", "jazz.00022.wav",
        "jazz.00042.wav", "jazz.00067.wav", "jazz.00060.wav", "jazz.00059.wav",
        "jazz.00065.wav", "jazz.00000.wav", "jazz.00030.wav", "jazz.00068.wav",
        "blues.00027.wav", "blues.00052.wav", "blues.00002.wav", "blues.00013.wav",
        "blues.00020.wav", "blues.00028.wav", "blues.00060.wav", "blues.00051.wav",
        "blues.00011.wav", "blues.00037.wav", "blues.00043.wav", "blues.00041.wav",
        "blues.00007.wav", "blues.00058.wav", "blues.00012.wav", "blues.00047.wav",
        "blues.00033.wav", "hiphop.00006.wav", "hiphop.00019.wav", "hiphop.00079.wav",
        "hiphop.00013.wav", "hiphop.00070.wav", "hiphop.00066.wav", "hiphop.00084.wav",
        "hiphop.00062.wav", "hiphop.00060.wav", "hiphop.00064.wav", "hiphop.00044.wav",
        "hiphop.00021.wav", "hiphop.00018.wav", "hiphop.00092.wav", "hiphop.00034.wav",
        "hiphop.00071.wav", "hiphop.00050.wav", "hiphop.00078.wav", "country.00028.wav",
        "country.00029.wav", "country.00033.wav", "country.00064.wav", "country.00056.wav",
        "country.00025.wav", "country.00068.wav", "country.00075.wav", "country.00032.wav",
        "country.00073.wav", "country.00087.wav", "country.00018.wav", "reggae.00023.wav",
        "reggae.00026.wav", "reggae.00083.wav", "reggae.00032.wav", "reggae.00044.wav",
        "reggae.00079.wav", "reggae.00019.wav", "reggae.00084.wav", "reggae.00070.wav",
        "reggae.00027.wav", "reggae.00008.wav", "reggae.00013.wav", "reggae.00012.wav",
        "reggae.00051.wav", "reggae.00074.wav", "reggae.00080.wav"



    ],


    "If the probability is above 0.8, listen to these songs.": [ "disco.00056.wav", "disco.00086.wav", "disco.00041.wav", "jazz.00034.wav", "jazz.00043.wav", 
        "jazz.00011.wav", "jazz.00025.wav", "jazz.00099.wav", "jazz.00015.wav", "jazz.00012.wav", 
        "jazz.00031.wav", "jazz.00062.wav", "jazz.00089.wav", "jazz.00064.wav", "jazz.00027.wav", 
        "jazz.00048.wav", "jazz.00076.wav", "jazz.00053.wav", "pop.00063.wav", "pop.00003.wav", 
        "pop.00040.wav", "blues.00000.wav", "blues.00035.wav", "blues.00001.wav", "blues.00004.wav", 
        "blues.00009.wav", "blues.00051.wav", "blues.00090.wav", "blues.00038.wav", "blues.00080.wav", 
        "blues.00057.wav", "blues.00029.wav", "blues.00008.wav", "blues.00034.wav", "blues.00056.wav", 
        "blues.00018.wav", "blues.00005.wav", "blues.00060.wav", "hiphop.00050.wav", "hiphop.00060.wav", 
        "hiphop.00061.wav", "hiphop.00055.wav", "hiphop.00057.wav", "hiphop.00013.wav", "hiphop.00096.wav", 
        "hiphop.00099.wav", "hiphop.00043.wav", "country.00096.wav", "country.00022.wav", "country.00090.wav", 
        "country.00057.wav", "country.00072.wav", "country.00010.wav", "country.00063.wav", "country.00046.wav", 
        "country.00047.wav", "country.00044.wav", "country.00012.wav", "country.00036.wav", "country.00059.wav", 
        "country.00054.wav", "country.00088.wav", "country.00024.wav", "country.00027.wav", "country.00050.wav", 
        "reggae.00067.wav", "reggae.00008.wav", "reggae.00025.wav", "reggae.00084.wav", "reggae.00077.wav", 
        "reggae.00017.wav", "reggae.00033.wav", "reggae.00002.wav", "reggae.00031.wav", "reggae.00024.wav", 
        "reggae.00063.wav", "reggae.00029.wav", "reggae.00001.wav", "reggae.00028.wav", "reggae.00035.wav", 
        "reggae.00038.wav", "reggae.00007.wav", "reggae.00085.wav", "reggae.00040.wav", "reggae.00026.wav", 
        "reggae.00019.wav", "reggae.00013.wav", "reggae.00014.wav", "reggae.00005.wav", "reggae.00094.wav", 
        "reggae.00037.wav", "reggae.00020.wav", "classical.00046.wav", "classical.00096.wav", "classical.00060.wav", 
        "classical.00098.wav", "classical.00095.wav", "jazz.00004.wav", "jazz.00071.wav", "jazz.00035.wav", "jazz.00044.wav", "jazz.00009.wav", 
        "jazz.00003.wav", "jazz.00010.wav", "jazz.00045.wav", "jazz.00002.wav", "jazz.00055.wav", 
        "blues.00091.wav", "blues.00089.wav", "blues.00092.wav", "blues.00021.wav", "blues.00093.wav", 
        "country.00067.wav", "country.00079.wav", "country.00065.wav", "classical.00023.wav", 
        "classical.00031.wav", "classical.00063.wav", "classical.00032.wav", "classical.00083.wav", 
        "classical.00065.wav", "classical.00078.wav", "classical.00058.wav", "classical.00027.wav", 
        "classical.00066.wav", "classical.00079.wav", "classical.00084.wav", "classical.00055.wav", 
        "classical.00043.wav", "classical.00011.wav", "classical.00035.wav", "classical.00087.wav", 
        "classical.00077.wav"]
}

@app.route("/result")
def result():
    return render_template("result.html", clusters=clustered_data)

@app.route("/search", methods=["GET"])
def search():
    # Render the "About Us" page
    return render_template("search.html")





# Path to the dataset (adjust this to your actual path)
dataset_path = "C:/Users/udara/Downloads/gtzann/genres_original"

# Static directory for serving audio
STATIC_DIR = 'static/audio'

# Function to search for a song based on genre and song name
def search_song(genre, song_name):
    song_path = os.path.join(dataset_path, genre, song_name)
    
    if os.path.exists(song_path):
        return song_path
    else:
        return None

@app.route("/song_search", methods=["GET", "POST"])
def song_search():
    if request.method == "POST":
        genre = request.form.get("genre")
        song_name = request.form.get("song_name")
        
        song_path = search_song(genre, song_name)
        if song_path:
            # Load the song with librosa
            audio, sample_rate = librosa.load(song_path, sr=None)
            
            # Copy the audio file to static directory to be served by Flask
            filename = secure_filename(song_name)
            shutil.copy(song_path, os.path.join(STATIC_DIR, filename))
            
            # Return the song search page with audio player
            return render_template("song_search.html", song_name=filename)
        else:
            return render_template("song_search.html", error="Song not found. Please try again.")
    
    return render_template("song_search.html")









@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["video"]
    
    # Check if the file is valid
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)
    
    # Log the file path to verify if the file is saved correctly
    print(f"File uploaded successfully: {filepath}")

    # Analyze the stress level and probability
    stress_level, probability = analyze_stress(filepath)

    # Convert probability to a native Python float (so it can be serialized to JSON)
    probability = float(probability)
    
    # Return stress level and probability as response
    return jsonify({"stress_level": stress_level, "probability": probability})

if __name__ == "__main__":
    app.run(debug=True)

