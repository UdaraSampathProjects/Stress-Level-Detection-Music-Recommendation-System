function uploadVideo() {
    let fileInput = document.getElementById("videoUpload");
    let file = fileInput.files[0];
    
    if (!file) {
        alert("Please select a video file.");
        return;
    }

    let formData = new FormData();
    formData.append("video", file);

    fetch("/upload", {
        method: "POST",
        body: formData,
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById("result").innerText = 
            `Stress Level: ${data.stress_level} (Probability: ${data.probability})`;
    })
    .catch(error => console.error("Error:", error));
}
