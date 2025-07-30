const form = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const resultsBox = document.getElementById("results");
const pdfViewer = document.getElementById("pdfViewer");
const summaryText = document.getElementById("summaryText");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  const response = await fetch("/upload/", {
    method: "POST",
    body: formData
  });

  const data = await response.json();
  pdfViewer.src = data.pdf_path;
  resultsBox.textContent = JSON.stringify(data.fields, null, 2);
  summaryText.textContent = "These extracted values can now be used to compute potential savings based on competitive rates and fees.";
});

function loadSample(path) {
  pdfViewer.src = path;
  resultsBox.textContent = "Click 'Upload' to analyze this sample.";
  summaryText.textContent = "";
}