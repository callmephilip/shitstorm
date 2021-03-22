var el = x => document.getElementById(x);

function showPicker() {
  el("file-input").click();
}

function showPicked(input) {
  var reader = new FileReader();
  reader.onload = function(e) {
    el("image-picked").src = e.target.result;
    el("image-picked").className = "";
  };
  reader.readAsDataURL(input.files[0]); 
}

function renderPredictions({ boxes }) {
  const imageCanvas = document.getElementById('image-canvas');
  boxes.forEach(({ xmin, xmax, ymin, ymax }) => {
    const box = document.createElement('div');
    box.setAttribute('style', `
      border: solid 1px hotpink;
      position: absolute; 
      left: ${xmin}; 
      top: ${ymin}; 
      width: ${xmax - xmin}px;
      height: ${ymax - ymin}px;
    `)
    imageCanvas.appendChild(box);
  });
}

function analyze() {
  var uploadFiles = el("file-input").files;
  if (uploadFiles.length !== 1) alert("Please select a file to analyze!");

  el("analyze-button").innerHTML = "Analyzing...";
  var xhr = new XMLHttpRequest();
  var loc = window.location;
  xhr.open("POST", `${loc.protocol}//${loc.hostname}:${loc.port}/analyze`,
    true);
  xhr.onerror = function() {
    alert(xhr.responseText);
  };
  xhr.onload = function(e) {
    if (this.readyState === 4) {
      renderPredictions(JSON.parse(e.target.responseText));
    }
    el("analyze-button").innerHTML = "Analyze";
  };

  var fileData = new FormData();
  fileData.append("file", uploadFiles[0]);
  xhr.send(fileData);
}

