// Live preview of uploaded image
document.addEventListener("DOMContentLoaded", function () {
  const fileInput = document.getElementById("file-upload");
  const preview = document.querySelector('.preview');
  const customFileUpload = document.querySelector('.custom-file-upload');
  const predictBtn = document.querySelector('.button.accent');
  // Live image preview
  if (fileInput && preview) {
    fileInput.addEventListener("change", function () {
      if (this.files && this.files[0]) {
        const reader = new FileReader();
        reader.onload = e => {
          preview.src = e.target.result;
          preview.style.display = "block";
          preview.classList.add("pop-in");
          setTimeout(() => preview.classList.remove("pop-in"), 600);
        };
        reader.readAsDataURL(this.files[0]);
      }
    });
  }
  // Animate file upload label on click
  if (customFileUpload) {
    customFileUpload.addEventListener('mousedown', () => {
      customFileUpload.classList.add('pressed');
    });
    customFileUpload.addEventListener('mouseup', () => {
      setTimeout(() => customFileUpload.classList.remove('pressed'), 120);
    });
    customFileUpload.addEventListener('mouseleave', () => {
      customFileUpload.classList.remove('pressed');
    });
  }
  // Predict button ripple animation
  if (predictBtn) {
    predictBtn.addEventListener('click', function(e){
      const ripple = document.createElement('span');
      ripple.className = 'ripple';
      ripple.style.left = `${e.offsetX}px`;
      ripple.style.top = `${e.offsetY}px`;
      this.appendChild(ripple);
      setTimeout(()=>ripple.remove(), 600);
    });
  }
  // Fade in result sections
  document.querySelectorAll('.result section').forEach(section => {
    section.classList.add('fade-section');
    setTimeout(() => section.classList.add('visible'), 350);
  });
  // Animate background orbs
  const orbLayers = document.querySelector('.glow-orbs');
  if (orbLayers) {
    let t = 0;
    setInterval(() => {
      t += 0.008;
      orbLayers.style.background =
        `radial-gradient(circle at ${25 + 5*Math.sin(t)}% ${25 + 4*Math.cos(t)}%, #3f51b5cc 120px, transparent 150px),` +
        `radial-gradient(circle at ${75 + 7*Math.cos(-t)}% ${40 + 6*Math.sin(t/2)}%, #7986cb99 140px, transparent 200px),` +
        `radial-gradient(circle at ${50 + 9*Math.sin(-t/1.4)}% ${75 + 5*Math.cos(t/1.2)}%, #4a67bd99 130px, transparent 180px)`;
    }, 60);
  }
});
