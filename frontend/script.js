document.addEventListener("DOMContentLoaded", () => {
  const fileInput = document.getElementById("videoFile");
  const fileNameDisplay = document.getElementById("fileName");
  const dropZone = document.getElementById("dropZone");
  const ppToggle = document.getElementById("postProcessingToggle");
  const ppSection = document.getElementById("postProcessingSection");
  const form = document.getElementById("transcribeForm");

  // File Upload UI
  fileInput.addEventListener("change", (e) => {
    if (e.target.files.length > 0) {
      fileNameDisplay.textContent = e.target.files[0].name;
      dropZone.classList.add("has-file");
    }
  });

  // Drag and Drop
  ["dragenter", "dragover", "dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(eventName, preventDefaults, false);
  });

  function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
  }

  ["dragenter", "dragover"].forEach((eventName) => {
    dropZone.addEventListener(
      eventName,
      () => dropZone.classList.add("dragover"),
      false
    );
  });

  ["dragleave", "drop"].forEach((eventName) => {
    dropZone.addEventListener(
      eventName,
      () => dropZone.classList.remove("dragover"),
      false
    );
  });

  dropZone.addEventListener("drop", (e) => {
    const dt = e.dataTransfer;
    const files = dt.files;
    fileInput.files = files;
    if (files.length > 0) {
      fileNameDisplay.textContent = files[0].name;
    }
  });

  // Post Processing Toggle
  function updatePostProcessingVisibility() {
    if (ppToggle.checked) {
      ppSection.classList.remove("hidden");
    } else {
      ppSection.classList.add("hidden");
    }
  }

  // Initial check
  updatePostProcessingVisibility();

  ppToggle.addEventListener("change", updatePostProcessingVisibility);

  // Form Submit
  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const file = fileInput.files[0];
    if (!file) {
      alert("Please select a video file.");
      return;
    }

    // Show loading
    form.classList.add("hidden");
    document.getElementById("loadingSection").classList.remove("hidden");

    const formData = new FormData();
    formData.append("video", file);
    formData.append("language", document.getElementById("language").value);
    formData.append(
      "whisper_prompt",
      document.getElementById("whisperPrompt").value
    );
    formData.append("post_processing", ppToggle.checked);
    formData.append(
      "post_processing_prompt",
      document.getElementById("postProcessingPrompt").value
    );

    try {
      const response = await fetch("http://localhost:8000/transcribe", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Transcription failed");
      }

      const data = await response.json();

      // Show result
      document.getElementById("loadingSection").classList.add("hidden");
      const resultSection = document.getElementById("resultSection");
      resultSection.classList.remove("hidden");

      const videoPlayer = document.getElementById("resultVideo");
      videoPlayer.src = "http://localhost:8000" + data.video_url;

      const downloadLink = document.getElementById("downloadTranscript");
      downloadLink.href = "http://localhost:8000" + data.transcript_url;
    } catch (error) {
      console.error(error);
      alert("An error occurred: " + error.message);
      // Reset UI
      document.getElementById("loadingSection").classList.add("hidden");
      form.classList.remove("hidden");
    }
  });

  // Back Button
  document.getElementById("backBtn").addEventListener("click", () => {
    document.getElementById("resultSection").classList.add("hidden");
    form.classList.remove("hidden");
    form.reset();
    fileNameDisplay.textContent = "Click or Drag Video File Here";
    dropZone.classList.remove("has-file");
    updatePostProcessingVisibility();
  });
});
