---
hide:
  - navigation
  - toc
---

<link rel="stylesheet" href="stylesheets/extra.css">

<!-- Hero Section -->
<section class="hero">
  <div class="hero-content">
    <h1 class="hero-title">Documentation that simply<br>STPP Benchmarking</h1>
    <p class="hero-description">
      A comprehensive framework for spatial-temporal point process benchmarking, 
      providing powerful tools and standardized metrics for evaluating predictive models.
    </p>
    <div class="hero-buttons">
      <a href="/setup/" class="btn btn-primary">Get Started</a>
      <a href="/Getting_started/Run_First_Experiment/run_first_experiment/" class="btn btn-secondary">Try a spidy experiment</a>
    </div>
  </div>
</section>

<section class="section-two" id="learn-more">
  <h2 class="section-title">
    The Most Advanced Framework for Spatial-Temporal Point Process Analysis
  </h2>
  <p class="section-subtitle">
    Built for researchers and practitioners who demand precision, scalability, 
    and reproducibility in their benchmarking workflows.
  </p>
</section>

<script>
document.addEventListener("DOMContentLoaded", function () {
  const main = document.querySelector(".md-main");
  if (main) {
    // Apply background image dynamically only for home.md
    main.style.backgroundImage = "url('../images/cover.svg')";
    main.style.backgroundSize = "cover";
    main.style.backgroundPosition = "center";
    main.style.backgroundRepeat = "no-repeat";
    main.style.backgroundAttachment = "fixed";

    // Optional dark overlay
    main.style.position = "relative";
    const overlay = document.createElement("div");
    overlay.style.position = "absolute";
    overlay.style.inset = "0";
    //overlay.style.background = "rgba(0, 0, 0, 0.6)";
    overlay.style.zIndex = "0";
    main.prepend(overlay);

    // Ensure content is above overlay
    const content = document.querySelector(".md-content");
    if (content) {
      content.style.position = "relative";
      content.style.zIndex = "1";
      content.style.color = "white";
    }
  }
});
</script>
