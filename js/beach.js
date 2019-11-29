var width = 0;
var height = 0;

var state = {
  resolution: 1000,
  lastWaveCreationTime: 0,
  waves: [
    {
      x:Math.random() * 500 + 400,
      dx: Math.random() * 0.8 - 0.6,
      h: 0,
      momentum: Math.random() * 15 + 20,
      spread: Math.random() * 15 + 5,
      opacity: 0.5,
    },
    {
      x:Math.random() * 500 + 400,
      dx: Math.random() * 0.8 - 0.6,
      h: 0,
      momentum: Math.random() * 15 + 20,
      spread: Math.random() * 15 + 5,
      opacity: 0.4,
    },
    {
      x:Math.random() * 500 + 400,
      dx: Math.random() * 0.8 - 0.6,
      h: 0,
      momentum: Math.random() * 15 + 20,
      spread: Math.random() * 15 + 5,
      opacity: 0.6,
    },
  ]
};

function poisson(mean) {
  var L = Math.exp(-mean);
  var p = 1.0;
  var k = 0;

  do {
    k++;
    p *= Math.random();
  } while (p > L);

  return k - 1;
}

function addWave() {
  state.waves.push({
    x:Math.random() * 500 + 400,
    dx: Math.random() * 0.8 - 0.6,
    h: 0,
    momentum: Math.random() * 15 + 20,
    spread: Math.random() * 15 + 5,
    opacity: Math.random() * 0.7,
  });
}

function physics() {
  for (var idx = 0; idx < state.waves.length; ++idx) {
    var wave = state.waves[idx];
    wave.spread += 0.01;
    wave.h += 0.01 * wave.momentum;
    wave.x += wave.dx;
    if (wave.h <= 0) {
      wave.h = 0;
      wave.momentum = 0;
    } else {
      wave.momentum -= 0.1;
    }
  }
  // How many waves should be created in the next 10 seconds?
  var d = new Date();
  var time = d.getTime();
  if (time - state.lastWaveCreationTime > 10000) {
    var wavesToCreate = poisson(7);
    var wavePeriod = 10000 / wavesToCreate;
    for (var i = 0; i < wavesToCreate; ++i) {
      var waveCreationTime = wavePeriod * i;
      setTimeout(addWave, waveCreationTime);
    }
    state.lastWaveCreationTime = d.getTime();
  }
}

function gaussianPlot(x, mean, deviation) {
  variance = deviation * deviation;
  return Math.pow(
      Math.exp(-(((x - mean) * (x - mean)) / ((2 * variance)))),
      1 / (deviation * Math.sqrt(2 * Math.PI)));
}

// create and fill polygon
CanvasRenderingContext2D.prototype.fillPolygon =
    function(pointsArray, fillColor, strokeColor) {
  if (pointsArray.length <= 0) return;
  this.moveTo(pointsArray[0][0], pointsArray[0][1]);
  for (var i = 0; i < pointsArray.length; i++) {
    this.lineTo(pointsArray[i][0], pointsArray[i][1]);
  }
  if (strokeColor != null && strokeColor != undefined)
    this.strokeStyle = strokeColor;

  if (fillColor != null && fillColor != undefined) {
    this.fillStyle = fillColor;
    this.fill();
  }
}

function drawWave(cvs, ctx, wave) {
  var firstX = 0;
  var firstY = cvs.height -
      gaussianPlot(0, wave.x, wave.spread) * wave.h / 100 * cvs.height - 10;
  var lastX = 0;
  var lastY = 0;
  ctx.beginPath();
  ctx.strokeStyle = 'rgba(0, 153, 255, 0.4)';
  ctx.moveTo(firstX, firstY);
  for (var point = 1; point <= state.resolution; ++point) {
    var x = (point / state.resolution) * cvs.width;
    var y = cvs.height -
        (gaussianPlot(x, wave.x, wave.spread)) * wave.h / 100 * cvs.height - 10;
    ctx.lineTo(x, y);
    lastX = x;
    lastY = y;
  }
  ctx.lineTo(lastX, cvs.height);
  ctx.lineTo(firstX, cvs.height);
  ctx.fillStyle = "rgba(61,184,180," + wave.opacity + ")";
  ctx.fill();
}

function render() {
  var canvas = document.getElementById('beach');
  var ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Draw waves.
  for (var idx = 0; idx < state.waves.length; ++idx) {
    drawWave(canvas, ctx, state.waves[idx]);
  }
}

function init() {
  /// get computed style for image
  var container = document.getElementById('beach-container');
  var cs = getComputedStyle(container);

  /// these will return dimensions in *pixel* regardless of what
  /// you originally specified for image:
  width = parseInt(cs.getPropertyValue('width'), 10);
  height = parseInt(cs.getPropertyValue('height'), 10);

  /// now use this as width and height for your canvas element:
  var canvas = document.getElementById('beach');

  canvas.width = width;
  canvas.height = height;
  window.requestAnimationFrame(loop);
}

function loop() {
  physics();
  render();
  window.requestAnimationFrame(loop);
}
