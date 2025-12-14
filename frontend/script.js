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

  // Form Submit (Placeholder)
  form.addEventListener("submit", (e) => {
    e.preventDefault();
    alert(
      "Transcribe started for: " +
        (fileInput.files[0] ? fileInput.files[0].name : "No file")
    );
  });
});
