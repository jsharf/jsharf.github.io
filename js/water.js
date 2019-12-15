// Water fluid dynamics inspired by this article:
// https://www.mikeash.com/pyblog/fluid-simulation-for-dummies.html
//
// I ported the algorithm to javascript and modified it to run in parallel on
// the GPU.
var state = {
  width: 0,
  height: 0,
  dt: 0.0,
  diff: 5.0,
  visc: 1,
  iterations: 1,

  s: [],
  density: [],

  vx: [],
  vy: [],

  vx0: [],
  vy0: [],
};

// Color Palette.
const r = [0x00, 0x5d, 0x96, 0xcb, 0x3f, 0xfd, 0xff, 0xfb, 0xe9];
const g = [0x42, 0x6a, 0x96, 0xc6, 0xff, 0xcd, 0x99, 0x5d, 0x00];
const b = [0x9d, 0xb3, 0xbf, 0xb5, 0x2c, 0x44, 0x4d, 0x4a, 0x2c];

function fill(arr, size, value) {
  for (var i = 0; i < size; ++i) {
    arr[i] = value;
  }
}

function Create2DArray(width, height) {
  var arr = [];

  for (var i = 0; i < width; i++) {
    arr[i] = [];
    fill(arr[i], height, 0);
  }

  return arr;
}

function drawDrop(frame, x, y, radius, value) {
  for (var i = x - radius; i < x + radius; ++i) {
    var x_circle = (i - x);
    var y_circle = Math.sqrt(radius * radius - x_circle * x_circle);
    var bottom = -y_circle + y;
    var top = y_circle + y;
    for (var j = bottom; j < top; ++j) {
      var indexI = Math.floor(i);
      var indexJ = Math.floor(j);
      frame[indexI][indexJ] = value;
    }
  }
}

function addWind(state, x, y, dirX, dirY) {
  for (var i = x - 10; i < x + 10; ++x) {
    for (var j = y - 10; j < y + 10; ++j) {
      state.vx0[i][j] = dirX;
      state.vy0[i][j] = dirY;
    }
  }
}

function initializeState(cvs) {
  const width = cvs.width;
  const height = cvs.height;
  state.width = width;
  state.height = height;
  state.s = Create2DArray(width, height);
  state.density = Create2DArray(width, height);
  state.vx = Create2DArray(width, height);
  state.vy = Create2DArray(width, height);
  state.vx0 = Create2DArray(width, height);
  state.vy0 = Create2DArray(width, height);
  drawDrop(state.s, width/2, 180, 20, 0.9);
  drawDrop(state.s, width/2, 140, 20, 0.9);
  addWind(state, width/2, 150, -10, 0);
}

function set_bnd(b, x, width, height) {
  // Constraint Y bounds
  for (var i = 1; i < width - 1; i++) {
    x[i][0] = b == 2 ? -x[i][1] : x[i][1];
    x[i][height - 1] =
        b == 2 ? -x[i][height - 2] : x[i][height - 2];
  }
  // Constrain X bounds
  for (var j = 1; j < height - 1; j++) {
    x[0][j] = b == 1 ? -x[1][j] : x[1][j];
    x[width - 1][j] = b == 1 ? -x[width - 2][j] : x[width - 2][j];
  }

  x[0][0] = 0.5 * (x[1][0] + x[0][1]);
  x[width - 1][height - 1] =
      0.5 * (x[width - 2][height - 1] + x[width - 1][height - 2]);
  x[0][height - 1] = 0.5 * (x[0][height - 2] + x[1][height - 1]);
  x[width - 1][0] = 0.5 * (x[width - 2][0] + x[width - 1][1]);
}

function lin_solve_parallel_iter(x, x0, a, c, width, height) {
  const i = this.thread.y;
  const j = this.thread.x;
  if ((i == 0) || (j == 0) || (i == width - 1) || (j == height - 1)) {
    return x[i][j];
  }
  const cRecip = 1.0 / c;
  return (x0[i][j] +
          a * (x[i + 1][j] + x[i - 1][j] + x[i][j + 1] + x[i][j - 1])) *
      cRecip;
}

var lin_solve_parallel_kernel;
function lin_solve_parallel(b, x, x0, a, c, iter, width, height) {
  if (typeof lin_solve_parallel_kernel == 'undefined') {
    lin_solve_parallel_kernel = gpu.createKernel(lin_solve_parallel_iter)
                                    .setGraphical(false)
                                    .setOutput([height, width])
                                    .setTactic('precision');
  }
  for (var i = 0; i < iter; ++i) {
    next_x = lin_solve_parallel_kernel(x, x0, a, c, width, height);
    const cRecip = 1.0 / c;
    set_bnd(b, next_x, width, height);
    x = next_x;
  }
  return x;
}

function diffuse_parallel(b, x, x0, diff, dt, iter, width, height) {
  // theory
  // Might need to remove one (N - 2) due to 3D -> 2D change.
  // var a = dt * diff * (N - 2) * (N - 2);
  // Jacob's theory is that this needs to be the boundary of the finite
  // volume. In 3D (commented out above), this was (N-2)^2. In 2D this is
  // (width + height) * 2
  var a = dt * diff * (2 * width + 2 * height);
  // theory
  // Might need to change 6 * a to 4 * a due to 3D -> 2D change.
  // lin_solve(b, x, x0, a, 1 + 6 * a, iter, width, height);
  return lin_solve_parallel(b, x, x0, a, 1 + 4 * a, iter, width, height);
}

var project_a_kernel;
var project_b_kernel;
function project_parallel(velocX, velocY, p, div, iter, width, height) {
  if (typeof project_a_kernel == 'undefined') {
    project_a_kernel =
        gpu.createKernel(function(velocX, velocY, width, height) {
             const i = this.thread.y;
             const j = this.thread.z;
             const divorp = this.thread.x;
             if ((i == 0) || (j == 0) || (i == width - 1) || (j == height - 1)) {
               return 0;
             }
             const div_component =
                 -0.5 * ((velocX[i + 1][j] - velocX[i - 1][j]) * width +
                         (velocY[i][j + 1] - velocY[i][j - 1]) * height);
             const p_component = 0;
             if (divorp == 0) {
               return div_component;
             } else {
               return p_component;
             }
           })
            .setTactic('speed')
            .setGraphical(false)
            .setOutput([height, width, 2]);
  }

  var divandp = project_a_kernel(velocX, velocY, width, height);
  div = divandp[0];
  p = divandp[1];
  set_bnd(0, div, width, height);
  // theory 6-> 4 for 3D -> 2D lin_solve(0, p, div, 1, 6, iter, N);
  p = lin_solve_parallel(0, p, div, 1, 4, iter, width, height);
  set_bnd(0, p, width, height);

  if (typeof project_b_kernel == 'undefined') {
    project_b_kernel =
        gpu.createKernel(function(velocX, velocY, p, width, height) {
             const i = this.thread.y;
             const j = this.thread.z;
             const velocXorY = this.thread.x;
             if ((i == 0) || (j == 0) || (i == width - 1) || (j == height - 1)) {
               return 0;
             }
             // theory. These used to be * N. Changed velocX to * width and
             // velocY to
             // * height. Alternatively, maybe X should be height and Y
             // should be width
             // since it's like measuring flux -- if you have a box and
             // particles are
             // traveling in the X direction, they'll hit the left or right
             // wall, not
             // the top or bottom, so the height of that wall would be
             // what's used to
             // measure avg flux through it..
             if (velocXorY == 0) {
               return velocX[i][j] - 0.5 * (p[i + 1][j] - p[i - 1][j]) * width;
             } else {
               return velocY[i][j] - 0.5 * (p[i][j + 1] - p[i][j - 1]) * height;
             }
           })
            .setTactic('speed')
            .setGraphical(false)
            .setOutput([height, width, 2]);
  }
  var velocXandY =
      project_b_kernel(velocX, velocY, p, width, height);
  velocX = velocXandY[0];
  velocY = velocXandY[1];
  set_bnd(1, velocX, width, height);
  set_bnd(2, velocY, width, height);
  return [velocX, velocY, p, div];
}

var advectKernel;
function advect_parallel(b, d, d0, velocX, velocY, dt, width, height) {
  if (typeof advectKernel == 'undefined') {
    advectKernel =
        gpu.createKernel(function(d0, velocX, velocY, dt, width, height) {
             const i = this.thread.y;
             const j = this.thread.x;
             // theory: what is going on here? units of time*space?
             // Used to be dtx = dty = dt * (N-2). Alternatively, just make
             // this dt. Or
             // swap x and Y. I dunno.
             const dtx = dt * (width - 2);
             const dty = dt * (height - 2);

             const offsetX = dtx * velocX[i][j];
             const offsetY = dty * velocY[i][j];
             const x = i - offsetX;
             const y = j - offsetY;

             if (x < 0.5) x = 0.5;
             if (x > width + 0.5) x = width + 0.5;
             const i0 = Math.floor(x);
             const i1 = i0 + 1.0;
             if (y < 0.5) y = 0.5;
             if (y > height + 0.5) y = height + 0.5;
             const j0 = Math.floor(y);
             const j1 = j0 + 1.0;

             const s1 = x - i0;
             const s0 = 1.0 - s1;
             const t1 = y - j0;
             const t0 = 1.0 - t1;

             const i0i = i0;
             const i1i = i1;
             const j0i = j0;
             const j1i = j1;

             return s0 * (t0 * d0[i0i][j0i]) + (t1 * d0[i0i][j1i]) +
                 s1 * (t0 * (d0[i1i][j0i]) + (t1 * (d0[i1i][j1i])))
           })
            .setGraphical(false)
            .setOutput([height, width])
            .setTactic('precision');
  }
  d = advectKernel(d0, velocX, velocY, dt, width, height);
  set_bnd(b, d, width, height);
  return d;
}

function physics() {
  console.log("Physics loop");
  state.vx0 = diffuse_parallel(
      1, state.vx0, state.vx, state.visc, state.dt, state.iterations, state.width,
      state.height);
  state.vy0 = diffuse_parallel(
      2, state.vy0, state.vy, state.visc, state.dt, state.iterations, state.width,
      state.height);

  var dpxy = project_parallel(
      state.vx0, state.vy0, state.vx, state.vy, state.iterations, state.width, state.height);
  state.vx0 = dpxy[0];
  state.vy0 = dpxy[1];
  state.vx = dpxy[2];
  state.vy = dpxy[3];


  state.vx = advect_parallel(
      1, state.vx, state.vx0, state.vx0, state.vy0, state.dt, state.width,
      state.height);
  state.vy = advect_parallel(
      2, state.vy, state.vy0, state.vx0, state.vy0, state.dt, state.width,
      state.height);

  dpxy = project_parallel(
      state.vx, state.vy, state.vx0, state.vy0, state.iterations, state.width, state.height);
  state.vx = dpxy[0];
  state.vy = dpxy[1];
  state.vx0 = dpxy[2];
  state.vy0 = dpxy[3];

  state.s = diffuse_parallel(
      0, state.s, state.density, state.diff, state.dt, state.iterations, state.width,
      state.height);
  state.density = advect_parallel(
      0, state.density, state.s, state.vx, state.vy, state.dt, state.width,
      state.height);
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

var renderKernel;
function render() {
  if (typeof renderKernel == 'undefined') {
    /// now use this as width and height for your canvas element:
    var canvas = document.getElementById('beach');
    const width = canvas.width;
    const height = canvas.height;
    const ctx = canvas.getContext('webgl2', {premultipliedAlpha: false});
    const renderSettings =
        {canvas: canvas, context: ctx, graphical: true, output: [width, height]};

    renderKernel = gpu.createKernel(function(r, g, b, densities, velocities) {
      const density = densities[this.thread.x][this.thread.y];
      const indexFloat = density * (8);
      if (indexFloat > 8) {
        indexFloat = 8;
      }
      const lowIndex = Math.floor(indexFloat);
      const highIndex = Math.ceil(indexFloat);
      const mix = indexFloat - lowIndex;

      const pr = (r[lowIndex] * mix + r[highIndex] * (1 - mix)) * 0.5;
      const pg = (g[lowIndex] * mix + g[highIndex] * (1 - mix)) * 0.5;
      const pb = (b[lowIndex] * mix + b[highIndex] * (1 - mix)) * 0.5;

      this.color(pr / 255, pg / 255, pb / 255);
    }, renderSettings);
  }

  renderKernel(r, g, b, state.density, state.vx);
}

window.onload = init;
function init() {
  /// get computed style for image
  var container = document.getElementById('beach-container');
  var cs = getComputedStyle(container);

  /// these will return dimensions in *pixel* regardless of what
  /// you originally specified for image:
  const width = parseInt(cs.getPropertyValue('width'), 10);
  const height = parseInt(cs.getPropertyValue('height'), 10);

  /// now use this as width and height for your canvas element:
  var canvas = document.getElementById('beach');

  canvas.width = width;
  canvas.height = height;

  initializeState(canvas);

  gpu = new GPU();
  console.log("Is GPU supported? " + GPU.isGPUSupported);
  console.log("Is WebGl2 supported? " + GPU.isGPUSupported);
  lastLoopDate = new Date();
  window.requestAnimationFrame(loop);
}

var lastLoopDate;
function loop() {
  // Calculate dt.
  var date = new Date();
  state.dt = (date.getTime() - lastLoopDate.getTime())/1000;
  if (state.dt == 0) {
    state.dt = 100;
  }
  lastLoopDate = date;

  physics();
  render();
  window.requestAnimationFrame(loop);
}
