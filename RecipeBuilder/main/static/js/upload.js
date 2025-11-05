// upload.js
document.addEventListener("DOMContentLoaded", function () {
  // --- Elements ---
  const uploadArea = document.getElementById("uploadArea");
  const imageUpload = document.getElementById("imageUpload");
  const previewContainer = document.getElementById("previewContainer");
  const imagePreview = document.getElementById("imagePreview");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const loadingContainer = document.getElementById("loadingContainer");
  const resultsContainer = document.getElementById("resultsContainer");
  const removeImage = document.getElementById("removeImage");
  const analyzeAnother = document.getElementById("analyzeAnother");
  const resultImage = document.getElementById("resultImage");

  // Django API endpoint
  const ANALYZE_ENDPOINT = "http://127.0.0.1:8000/api/analyze-food/";

  let currentImageFile = null;

  // --- CSRF helper ---
  function getCookie(name) {
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);
    if (parts.length === 2) return parts.pop().split(";").shift();
    return null;
  }
  const csrftoken = getCookie("csrftoken");

  // --- UI helpers ---
  function showPreview(src) {
    imagePreview.src = src;
    previewContainer.style.display = "block";
    uploadArea.style.display = "none";
    resultsContainer.style.display = "none";
    loadingContainer.style.display = "none";
  }

  function resetToUpload() {
    previewContainer.style.display = "none";
    uploadArea.style.display = "block";
    imageUpload.value = "";
    currentImageFile = null;
    resultsContainer.style.display = "none";
  }

  function showLoading() {
    previewContainer.style.display = "none";
    loadingContainer.style.display = "block";
    resultsContainer.style.display = "none";
  }

  function showResults() {
    loadingContainer.style.display = "none";
    resultsContainer.style.display = "block";
  }

  // --- File handling ---
  uploadArea.addEventListener("click", () => imageUpload.click());

  imageUpload.addEventListener("change", function () {
    if (this.files && this.files[0]) {
      const file = this.files[0];
      if (!file.type.match("image.*")) {
        alert("Please select an image file");
        return;
      }
      if (file.size > 5 * 1024 * 1024) {
        alert("File size must be less than 5MB");
        return;
      }

      currentImageFile = file;
      const reader = new FileReader();
      reader.onload = (e) => showPreview(e.target.result);
      reader.readAsDataURL(file);
    }
  });

  // Drag & drop support
  uploadArea.addEventListener("dragover", (e) => {
    e.preventDefault();
    uploadArea.classList.add("dragover");
  });
  uploadArea.addEventListener("dragleave", () => {
    uploadArea.classList.remove("dragover");
  });
  uploadArea.addEventListener("drop", function (e) {
    e.preventDefault();
    uploadArea.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      imageUpload.files = e.dataTransfer.files;
      imageUpload.dispatchEvent(new Event("change"));
    }
  });

  removeImage.addEventListener("click", resetToUpload);
  analyzeAnother.addEventListener("click", resetToUpload);

  // --- Analyze Button ---
  analyzeBtn.addEventListener("click", async function () {
    if (!currentImageFile) {
      alert("Please select an image first");
      return;
    }

    analyzeBtn.disabled = true;
    analyzeBtn.textContent = "Analyzing...";
    showLoading();

    try {
      const data = await analyzeFood(currentImageFile);
      resultImage.src = imagePreview.src;
      populateRecipeData(data);
      showResults();
      resultsContainer.scrollIntoView({ behavior: "smooth" });
    } catch (err) {
      console.error(err);
      alert("Error: " + (err.message || "Failed to analyze image."));
      previewContainer.style.display = "block";
    } finally {
      analyzeBtn.disabled = false;
      analyzeBtn.textContent = "Analyze Food";
    }
  });

  // --- Send image to Django API ---
  async function analyzeFood(imageFile) {
    const formData = new FormData();
    formData.append("image", imageFile);
    formData.append("get_recipe", "true");

    const resp = await fetch(ANALYZE_ENDPOINT, {
      method: "POST",
      body: formData,
      headers: { "X-CSRFToken": csrftoken },
      credentials: "same-origin",
    });

    if (!resp.ok) {
      let errText = `HTTP ${resp.status}`;
      try {
        const j = await resp.json();
        errText = j.error || j.message || JSON.stringify(j);
      } catch (e) {}
      throw new Error(errText);
    }
    return await resp.json();
  }

  // --- Render multiple categories and recipes ---
  function populateRecipeData(apiResponse) {
    resultsContainer.innerHTML = ""; // Clear previous results

    // Backend may return an array of recommendations
    const recommendations = Array.isArray(apiResponse)
      ? apiResponse
      : [apiResponse];

    if (recommendations.length === 0) {
      resultsContainer.innerHTML = "<p>No recipe details found.</p>";
      return;
    }

    recommendations.forEach((rec, recIndex) => {
      const category = rec.category || `Category ${recIndex + 1}`;

      // Handle unknown category
      if (category.toLowerCase() === "unknown") {
        const unknownDiv = document.createElement("div");
        unknownDiv.classList.add("recipe-card");
        unknownDiv.innerHTML =
          "<h3>Unknown Food</h3><p>We could not recognize this food.</p>";
        resultsContainer.appendChild(unknownDiv);
        return; // Skip rendering recipes for this category
      }
      const recipes = rec.recipes || [];

      recipes.forEach((r, rIndex) => {
        // Create a container for each recipe
        const recipeDiv = document.createElement("div");
        recipeDiv.classList.add("recipe-card");
        recipeDiv.style.marginBottom = "30px";

        // Recipe title
        const title = document.createElement("h3");
        title.textContent = r.name || category;
        recipeDiv.appendChild(title);

        // Only cooking time
        const cookTimeDiv = document.createElement("div");
        cookTimeDiv.classList.add("food-meta");
        const cookTimeSpan = document.createElement("span");
        cookTimeSpan.classList.add("meta-item");
        cookTimeSpan.innerHTML = `<i class="fas fa-clock"></i> ${
          r.cooking_time || r.time || "—"
        }`;
        cookTimeDiv.appendChild(cookTimeSpan);
        recipeDiv.appendChild(cookTimeDiv);

        // Ingredients
        const ingrDiv = document.createElement("div");
        ingrDiv.classList.add("ingredients-list");
        const ingrTitle = document.createElement("h4");
        ingrTitle.innerHTML = `<i class="fas fa-list"></i> Ingredients`;
        ingrDiv.appendChild(ingrTitle);

        const ingrUl = document.createElement("ul");
        const ingrArr = Array.isArray(r.ingredients)
          ? r.ingredients
          : typeof r.ingredients === "string"
          ? r.ingredients.split(/\n|,/)
          : [];

        ingrArr.forEach((item) => {
          if (!item) return;
          // Remove leading numbers if any
          const cleanItem = item.replace(/^\s*\d+[\.\)]\s*/, "").trim();
          const li = document.createElement("li");
          li.textContent = cleanItem;
          ingrUl.appendChild(li);
        });
        ingrDiv.appendChild(ingrUl);
        recipeDiv.appendChild(ingrDiv);

        // Steps
        const stepsDiv = document.createElement("div");
        stepsDiv.classList.add("steps-list");
        const stepsTitle = document.createElement("h4");
        stepsTitle.innerHTML = `<i class="fas fa-mortar-pestle"></i> Cooking Instructions`;
        stepsDiv.appendChild(stepsTitle);

        const stepsOl = document.createElement("ol");
        const stepsArr = Array.isArray(r.directions)
          ? r.directions
          : Array.isArray(r.instructions)
          ? r.instructions
          : typeof r.directions === "string"
          ? r.directions.split(/\n/)
          : [];

        stepsArr.forEach((step) => {
          if (!step) return;
          // Remove leading numbers
          const cleanStep = step.replace(/^\s*\d+[\.\)]\s*/, "").trim();
          const li = document.createElement("li");
          li.textContent = cleanStep;
          stepsOl.appendChild(li);
        });
        stepsDiv.appendChild(stepsOl);
        recipeDiv.appendChild(stepsDiv);

        resultsContainer.appendChild(recipeDiv);
      });
    });

    // Scroll to results
    resultsContainer.scrollIntoView({ behavior: "smooth" });
  }
});
