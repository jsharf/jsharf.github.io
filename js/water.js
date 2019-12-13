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
  visc: 5.0,

  s: [],
  density: [],

  vx: [],
  vy: [],

  vx0: [],
  vy0: [],
};

function fill(arr, size, value) {
  for (var i = 0; i < size; ++i) {
    arr[i] = value;
  }
}

function initializeState(cvs) {
  state.width = cvs.width;
  state.height = cvs.height;
  fill(state.s, state.width * state.height, 0);
  fill(state.density, state.width * state.height, 0);
  fill(state.vx, state.width * state.height, 0);
  fill(state.vy, state.width * state.height, 0);
  fill(state.vx0, state.width * state.height, 0);
  fill(state.vy0, state.width * state.height, 0);
}

function IX(x, y) {
  return x * state.width + y;
}

function set_bnd(b, x, width, height) {
  // Constraint Y bounds
  for (var i = 1; i < width - 1; i++) {
    x[IX(i, 0)] = b == 2 ? -x[IX(i, 1)] : x[IX(i, 1)];
    x[IX(i, height - 1)] =
        b == 2 ? -x[IX(i, height - 2)] : x[IX(i, height - 2)];
  }
  // Constrain X bounds
  for (var j = 1; j < height - 1; j++) {
    x[IX(0, j)] = b == 1 ? -x[IX(1, j)] : x[IX(1, j)];
    x[IX(width - 1, j)] = b == 1 ? -x[IX(width - 2, j)] : x[IX(width - 2, j)];
  }

  x[IX(0, 0)] = 0.5 * (x[IX(1, 0)] + x[IX(0, 1)]);
  x[IX(width - 1, height - 1)] =
      0.5 * (x[IX(width - 2, height - 1)] + x[IX(width - 1, height - 2)]);
  x[IX(0, height - 1)] = 0.5 * (x[IX(0, height - 2)] + x[IX(1, height - 1)]);
  x[IX(width - 1, 0)] = 0.5 * (x[IX(width - 2, 0)] + x[IX(width - 1, 1)]);
}

var set_bnd_y_kernel;
function set_bnd_y_parallel(b, x, width, height) {
  if (typeof set_bnd_y_kernel == 'undefined') {
    set_bnd_y_kernel =
        gpu.createKernel(function set_bnd_y(b, x, width, height) {
             var i = this.thread.x;
             x[IX(i, 0)] = b == 2 ? -x[IX(i, 1)] : x[IX(i, 1)];
             x[IX(i, height - 1)] =
                 b == 2 ? -x[IX(i, height - 2)] : x[IX(i, height - 2)];
           })
            .setGraphical(false)
            .setOutput([width]);
  }
  return set_bnd_y_kernel(b, x, width, height);
}

function set_bnd_corners(b, x, width, height) {
  x[IX(0, 0)] = 0.5 * (x[IX(1, 0)] + x[IX(0, 1)]);
  x[IX(width - 1, height - 1)] =
      0.5 * (x[IX(width - 2, height - 1)] + x[IX(width - 1, height - 2)]);
  x[IX(0, height - 1)] = 0.5 * (x[IX(0, height - 2)] + x[IX(1, height - 1)]);
  x[IX(width - 1, 0)] = 0.5 * (x[IX(width - 2, 0)] + x[IX(width - 1, 1)]);
}

var set_bnd_x_kernel;
function set_bnd_x_parallel(b, x, width, height) {
  if (typeof set_bnd_x_kernel == 'undefined') {
    set_bnd_x_kernel =
        gpu.createKernel(function set_bnd_y(b, x, width, height) {
             var j = this.thread.x;
             x[IX(0, j)] = b == 1 ? -x[IX(1, j)] : x[IX(1, j)];
             x[IX(width - 1, j)] =
                 b == 1 ? -x[IX(width - 2, j)] : x[IX(width - 2, j)];
           })
            .setGraphical(false)
            .setOutput([height]);
  }
  set_bnd_x_kernel(b, x, width, height);
}

function set_bnd_parallel(b, x, width, height) {
  set_bnd_x_parallel(b, x, width, height);
  set_bnd_y_parallel(b, x, width, height);
  set_bnd_corners(b, x, width, height);
}

function lin_solve(b, x, x0, a, c, iter, width, height) {
  var cRecip = 1.0 / c;
  for (var k = 0; k < iter; k++) {
    for (var j = 1; j < height - 1; j++) {
      for (var i = 1; i < width - 1; i++) {
        x[IX(i, j)] = (x0[IX(i, j)] +
                       a * (x[IX(i + 1, j)] + x[IX(i - 1, j)] +
                            x[IX(i, j + 1)] + x[IX(i, j - 1)])) *
            cRecip;
      }
    }
    set_bnd(b, x, width, height);
  }
}

function diffuse(b, x, x0, diff, dt, iter, width, height) {
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
  lin_solve(b, x, x0, a, 1 + 4 * a, iter, width, height);
}

function lin_solve_parallel_iter(b, x, x0, a, c, iter, width, height) {
  var i = this.thread.x + 1;
  var j = this.thread.y + 1;
  var cRecip = 1.0 / c;
  x[IX(i, j)] = (x0[IX(i, j)] +
                 a * (x[IX(i + 1, j)] + x[IX(i - 1, j)] + x[IX(i, j + 1)] +
                      x[IX(i, j - 1)])) *
      cRecip;
}

var lin_solve_parallel_kernel;
function lin_solve_parallel(b, x, x0, a, c, iter, width, height) {
  if (typeof lin_solve_parallel_iter == 'undefined') {
    gpu.addFunction(set_bnd_at);
    lin_solve_parallel_kernel = gpu.createKernel(lin_solve_parallel_iter)
                                    .setOutput([width - 2, height - 2])
                                    .setGraphical(false);
  }
  for (var k = 0; i < iter; ++k) {
    lin_solve_parallel_kernel(b, x, x0, a, c, iter, width, height);
    set_bnd_parallel(b, x, width, height);
  }
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
  lin_solve_parallel(b, x, x0, a, 1 + 4 * a, iter, width, height);
}

var project_a_kernel;
var project_b_kernel;
function project_parallel(velocX, velocY, velocZ, p, div, iter, width, height) {
  if (typeof project_a_kernel == 'undefined') {
    project_a_kernel = gpu.createKernel(function(velocX, velocY, velocZ, p, div, iter, width, height) {
      var i = this.thread.x + 1;
      var j = this.thread.y + 1;
      div[IX(i, j)] =
          -0.5 * ((velocX[IX(i + 1, j)] - velocX[IX(i - 1, j)]) / width +
                  (velocY[IX(i, j + 1)] - velocY[IX(i, j - 1)]) / height);
      p[IX(i, j)] = 0;
    }).setOutput([width-2, height-2]).setGraphical(false);
  }

  project_a_kernel(velocX, velocY, velocZ, p, div, iter, width, height);
  set_bnd_parallel(0, div, width, height);
  set_bnd_parallel(0, p, width, height);
  // theory 6-> 4 for 3D -> 2D lin_solve(0, p, div, 1, 6, iter, N);
  lin_solve_parallel(0, p, div, 1, 4, iter, width, height);

  if (typeof project_b_kernel == 'undefined') {
    project_b_kernel = gpu.createKernel(function(velocX, velocY, velocZ, p, div, iter, width, height) {
      var i = this.thread.x + 1;
      var j = this.thread.y + 1;
      // theory. These used to be * N. Changed velocX to * width and velocY to
      // * height. Alternatively, maybe X should be height and Y should be width
      // since it's like measuring flux -- if you have a box and particles are
      // traveling in the X direction, they'll hit the left or right wall, not
      // the top or bottom, so the height of that wall would be what's used to
      // measure avg flux through it..
      velocX[IX(i, j)] -= 0.5 * (p[IX(i + 1, j)] - p[IX(i - 1, j)]) * width;
      velocY[IX(i, j)] -= 0.5 * (p[IX(i, j + 1)] - p[IX(i, j - 1)]) * height;
    }).setOutput([width-2, height-2]).setGraphical(false);
  }
  project_b_kernel(velocX, velocY, velocZ, p, div, iter, width, height);
  set_bnd_parallel(1, velocX, width, height);
  set_bnd_parallel(2, velocY, width, height);
}

var advectKernel;
function advect_parallel(b, d, d0, velocX, velocY, velocZ, dt, width, height) {
  if (typeof advectKernel == 'undefined') {
    advectKernel = gpu.createKernel(function(b, d, d0, velocX, velocY, velocZ, dt, width, height) {
      var i = this.thread.x + 1;
      var j = this.thread.y + 1;
      // theory: what is going on here? units of time*space?
      // Used to be dtx = dty = dt * (N-2). Alternatively, just make this dt. Or
      // swap x and Y. I dunno.
      var dtx = dt * (width - 2);
      var dty = dt * (height - 2);

      var s0, s1, t0, t1, u0, u1;

      var offsetX = dtx * velocX[IX(i, j)];
      var offsetY = dty * velocY[IX(i, j)];
      var x = i - offsetX;
      var y = j - offsetY;

      if (x < 0.5) x = 0.5;
      if (x > width + 0.5) x = width + 0.5;
      var i0 = Math.floor(x);
      var i1 = i0 + 1.0;
      if (y < 0.5) y = 0.5;
      if (y > height + 0.5) y = height + 0.5;
      var j0 = Math.floor(y);
      var j1 = j0 + 1.0;

      s1 = x - i0;
      s0 = 1.0 - s1;
      t1 = y - j0;
      t0 = 1.0 - t1;

      var i0i = i0;
      var i1i = i1;
      var j0i = j0;
      var j1i = j1;

      d[IX(i, j)] = s0 * (t0 * d0[IX(i0i, j0i)]) + (t1 * d0[IX(i0i, j1i)]) +
          s1 * (t0 * (d0[IX(i1i, j0i)]) + (t1 * (d0[IX(i1i, j1i)])))
    }).setOutput([width - 2, height - 2]).setGraphical(false);
  }
  advectKernel();
  set_bnd_parallel(b, d, width, height);
}

function physics() {
  diffuse_parallel(
      1, state.vx0, state.vx, state.visc, state.dt, 4, state.width,
      state.height);
  diffuse_parallel(
      2, state.vy0, state.vy, state.visc, state.dt, 4, state.width,
      state.height);

  project_parallel(
      state.vx0, state.vy0, state.vx, state.vy, 4, state.width, state.height);

  advect_parallel(
      1, state.vx, state.vx0, state.vx0, state.vy0, state.dt, state.width,
      state.height);
  advect_parallel(
      2, state.vy, state.vy0, state.vx0, state.vy0, state.dt, state.width,
      state.height);

  project_parallel(
      state.vx, state.vy, state.vx0, state.vy0, 4, state.width, state.height);

  diffuse_parallel(
      0, state.s, state.density, state.diff, state.dt, 4, state.width,
      state.height);
  advect_parallel(
      0, state.density, state.s, state.vx, state.vy, state.dt, state.width,
      state.height);
}

// Maps value from 0 -> 1 through a diverging color palette.
function color(n) {
  var r = [0x00, 0x5d, 0x96, 0xcb, 0x3f, 0xfd, 0xff, 0xfb, 0xe9];
  var g = [0x42, 0x6a, 0x96, 0xc6, 0xff, 0xcd, 0x99, 0x5d, 0x00];
  var b = [0x9d, 0xb3, 0xbf, 0xb5, 0x2c, 0x44, 0x4d, 0x4a, 0x2c];
  var indexFloat = n * r.size;
  var lowIndex = Math.floor(indexFloat);
  var highIndex = Math.ceil(indexFloat);
  var mix = indexFloat - lowIndex;

  var color = [];
  // r
  color[0] = (r[lowIndex] * mix + r[highIndex] * (1 - mix)) * 0.5;
  // g
  color[1] = (g[lowIndex] * mix + g[highIndex] * (1 - mix)) * 0.5;
  // b
  color[2] = (b[lowIndex] * mix + b[highIndex] * (1 - mix)) * 0.5;
  return color;
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
  var canvas = document.getElementById('beach');
  var ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (typeof renderKernel == 'undefined') {
    console.log("renderKernel undefined!");
    return;
  }

  renderKernel();
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

  initializeState(canvas);

  gpu = new GPU();
  var renderSettings = {
    canvas: canvas,
    output: {x: width, y: height},
  };

  renderKernel =
      gpu.createKernel(function() {
           var density = state.density[IX(this.thread.x, this.thread.y)];
           var color = color(density);
           this.color(color[0], color[1], color[2], 1);
         }, renderSettings).setGraphical(true).setOutput([width, height]);
  lastLoopDate = new Date();
  window.requestAnimationFrame(loop);
}

var lastLoopDate;
function loop() {
  // Calculate dt.
  var date = new Date();
  state.dt = (date.getTime() - lastLoopDate.getTime()) / 1000;
  lastLoopDate = date;

  physics();
  render();
  window.requestAnimationFrame(loop);
}
