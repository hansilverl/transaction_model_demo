const form = document.getElementById("uploadForm");
const fileInput = document.getElementById("fileInput");
const resultsBox = document.getElementById("results");
const pdfViewer = document.getElementById("pdfViewer");
const summaryText = document.getElementById("summaryText");
const resultsPanel = document.getElementById("resultsBox");

async function analyzeCurrentFile() {
  if (!fileInput.files.length) {
    resultsBox.textContent = "No file selected.";
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  const response = await fetch("/upload/", {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const err = await response.text();
    resultsBox.textContent = `Error: ${err}`;
    pdfViewer.src = "";
    summaryText.textContent = "";
    return;
  }

  const data = await response.json();
  pdfViewer.src = data.pdf_path;
  resultsBox.textContent = JSON.stringify(data.fields, null, 2);
  summaryText.textContent =
    "These fields can be translated and used in your application to calculate how much you can save.";

  if (window.innerWidth <= 768) {
    resultsPanel.scrollIntoView({ behavior: "smooth" });
  }
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  await analyzeCurrentFile();
});

async function loadSample(path) {
  pdfViewer.src = path;
  resultsBox.textContent = "Analyzing sample, please wait...";
  summaryText.textContent = "";

  const res = await fetch(path);
  const blob = await res.blob();
  const file = new File([blob], path.split("/").pop(), {
    type: "application/pdf",
  });
  const dt = new DataTransfer();
  dt.items.add(file);
  fileInput.files = dt.files;

  await analyzeCurrentFile();
}
