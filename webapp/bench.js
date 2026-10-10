/* The live bench: two trained networks running in the browser, five reductions on top of them.
 *
 * Project : CISC3024 AI Assignment #2 (Pattern Recognition, University of Macau)
 * Author  : AI-generated implementation (Qoder). The student wrote no code.
 *
 * The graphs come from src/export_onnx.py and are the trained models with spectral normalisation
 * folded into plain weights. Everything after the backbone -- the flexible-Dirichlet moments, the
 * temperature, the Dirichlet rescaling, the Laplace weight draws -- is recomputed here from
 * data/instruments.json using the same arithmetic as src/fd_distribution.py, src/calibrate.py and
 * src/laplace_icla.py. The "check this page against the report" control exists because that claim
 * should be testable by a visitor rather than taken on trust.
 *
 * Register grammar: hue marks the instrument, line form marks its state. A struck rule always means
 * "this quantity does not exist here", never "this quantity measured zero".
 */

(function () {
  "use strict";

  var status = document.getElementById("bench-status");
  var runtimeField = document.getElementById("field-runtime");
  var specimenField = document.getElementById("field-specimen");

  var INSTR = null;      // data/instruments.json
  var SESS = {};         // baseline / evidential onnxruntime sessions
  /* One inference at a time: the two WebGPU sessions share a device, and overlapping runs on the
     same session throw "Session already started". A visitor dragging the brush produces runs far
     faster than the network can answer them, so requests are chained rather than raced. */
  var CHAIN = Promise.resolve();

  /* ---------------------------------------------------------------- math */

  function softmax(v) {
    var m = -Infinity, i, s = 0, out = new Array(v.length);
    for (i = 0; i < v.length; i++) if (v[i] > m) m = v[i];
    for (i = 0; i < v.length; i++) { out[i] = Math.exp(v[i] - m); s += out[i]; }
    for (i = 0; i < v.length; i++) out[i] /= s;
    return out;
  }

  /* The flexible-Dirichlet moments, ported line for line from fd_distribution.py. */
  function fdMoments(alpha, p, tau) {
    var K = alpha.length, i;
    var alpha0 = 0;
    for (i = 0; i < K; i++) alpha0 += alpha[i];
    var denom = alpha0 + tau;
    var mu = new Array(K), sum = 0;
    for (i = 0; i < K; i++) { mu[i] = (alpha[i] + tau * p[i]) / denom; sum += mu[i]; }
    for (i = 0; i < K; i++) mu[i] /= Math.max(sum, 1e-12);
    var varr = new Array(K);
    for (i = 0; i < K; i++) {
      var pi = p[i];
      varr[i] = mu[i] * (1 - mu[i]) / (denom + 1) +
        (tau * tau) * pi * (1 - pi) / (denom * (denom + 1));
    }
    var epistemic = 0, total = 0;
    for (i = 0; i < K; i++) { epistemic += varr[i]; total += mu[i] * mu[i]; }
    total = 1 - total;
    return { mean: mu, variance: varr, epistemic: epistemic, total: total,
             aleatoric: total - epistemic, alpha0: alpha0, tau: tau, alpha: alpha, p: p };
  }

  /* Deterministic pseudo-random draws, so the Laplace panel gives the same answer on reload. */
  function gaussian(seed) {
    var state = seed >>> 0;
    function unit() {
      state |= 0; state = (state + 0x6D2B79F5) | 0;
      var t = Math.imul(state ^ (state >>> 15), 1 | state);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
    return function () {
      var u = Math.max(unit(), 1e-12), v = unit();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    };
  }

  /* ---------------------------------------------------------------- state */

  var canvas = document.getElementById("specimen");
  var ctx = canvas.getContext("2d", { willReadFrequently: true });
  var brush = 1, erasing = false;

  function clearPlate(white) {
    ctx.fillStyle = white ? "#ffffff" : "#000000";
    ctx.fillRect(0, 0, 32, 32);
    ctx.globalCompositeOperation = "source-over";
  }
  clearPlate(true);

  function stamp(x, y) {
    ctx.fillStyle = erasing ? "#ffffff" : "#000000";
    var r = brush;
    ctx.fillRect(Math.round(x) - (r >> 1), Math.round(y) - (r >> 1), r, r);
  }

  function pointerToPixel(evt) {
    var box = canvas.getBoundingClientRect();
    var t = evt.touches ? evt.touches[0] : evt;
    return [Math.floor((t.clientX - box.left) / box.width * 32),
            Math.floor((t.clientY - box.top) / box.height * 32)];
  }

  var drawing = false;
  function onDown(e) { drawing = true; move(e, true); e.preventDefault(); }
  function onUp() { if (drawing) { drawing = false; run(); } }
  function move(e, force) {
    if (!drawing && !force) return;
    var xy = pointerToPixel(e);
    stamp(xy[0], xy[1]);
    if (force || e.type !== "pointermove") run();
    e.preventDefault();
  }
  canvas.addEventListener("pointerdown", onDown);
  canvas.addEventListener("pointermove", function (e) { move(e, false); });
  window.addEventListener("pointerup", onUp);

  function imageToPlate(img) {
    var tmp = document.createElement("canvas");
    tmp.width = 32; tmp.height = 32;
    var c = tmp.getContext("2d");
    c.imageSmoothingEnabled = true;
    c.drawImage(img, 0, 0, 32, 32);
    ctx.clearRect(0, 0, 32, 32);
    ctx.drawImage(tmp, 0, 0);
  }

  /* ---------------------------------------------------------------- inference */

  function tensorFromPlate() {
    var d = ctx.getImageData(0, 0, 32, 32).data;
    var out = new Float32Array(3 * 32 * 32);
    var mean = INSTR.preprocessing.mean, std = INSTR.preprocessing.std;
    for (var i = 0; i < 32 * 32; i++) {
      for (var ch = 0; ch < 3; ch++) {
        out[ch * 1024 + i] = (d[i * 4 + ch] / 255 - mean[ch]) / std[ch];
      }
    }
    return new ort.Tensor("float32", out, [1, 3, 32, 32]);
  }

  function reduceAll(logits, features, alpha, p, tau) {
    var K = logits.length, i, j;
    var out = {};

    out.B1_ce = (function () {
      var m = softmax(logits);
      return summarise(m, zeros(K), 0, m);
    })();

    out.B2_ce_temperature = (function () {
      var T = Math.max(INSTR.temperature.T, 1e-3);
      var m = softmax(logits.map(function (v) { return v / T; }));
      return summarise(m, zeros(K), 0, m);
    })();

    var fd = fdMoments(alpha, p, tau);
    out.A1_fedl = summarise(fd.mean, fd.variance, fd.epistemic, fd.mean);
    out.A1_internals = fd;

    out.A2_fedl_dirichlet = (function () {
      var t = INSTR.dirichlet_scaling.t, b = INSTR.dirichlet_scaling.b;
      var z = fd.mean.map(function (mu, k) {
        return t[k] * Math.log(Math.max(mu, 1e-8)) + b[k];
      });
      var m = softmax(z);
      return summarise(m, zeros(K), 0, m);
    })();

    out.A3_laplace_icla = (function () {
      var W = INSTR.laplace.W_map, bmap = INSTR.laplace.b_map, sigma = INSTR.laplace.posterior_std;
      var S = INSTR.laplace.n_samples, d = features.length;
      var rand = gaussian(INSTR.seed);
      var acc = zeros(K), acc2 = new Array(K * S);
      for (var s = 0; s < S; s++) {
        var lg = new Array(K);
        for (var k = 0; k < K; k++) {
          var row = W[k], sum = bmap[k];
          for (var q = 0; q < d; q++) sum += (row[q] + sigma[k * d + q] * rand()) * features[q];
          lg[k] = sum;
        }
        var pr = softmax(lg);
        for (k = 0; k < K; k++) { acc[k] += pr[k]; acc2[s * K + k] = pr[k]; }
      }
      var mean = acc.map(function (v) { return v / S; });
      var varr = new Array(K);
      for (var k2 = 0; k2 < K; k2++) {
        var e = 0;
        for (var s2 = 0; s2 < S; s2++) { var dz = acc2[s2 * K + k2] - mean[k2]; e += dz * dz; }
        varr[k2] = e / S;
      }
      return summarise(mean, varr, varr.reduce(add, 0), mean);
    })();

    return out;
  }
  function zeros(n) { return new Array(n).fill(0); }
  function add(a, b) { return a + b; }
  function summarise(mean, variance, epistemic) {
    var total = 1 - mean.reduce(function (s, v) { return s + v * v; }, 0);
    var best = 0;
    for (var i = 1; i < mean.length; i++) if (mean[i] > mean[best]) best = i;
    return { mean: mean, variance: variance, epistemic: epistemic, total: total,
             aleatoric: total - epistemic, predicted: best, confidence: mean[best] };
  }

  function run() {
    if (!INSTR || !SESS.baseline) return;
    CHAIN = CHAIN.then(infer).catch(function (err) {
      setStatus("error", "The instrument failed on this specimen: " + (err.message || err) +
        ". Reload the page; if it persists, the model files in <code>model/</code> are not the ones " +
        "written by <code>src/export_onnx.py</code>.");
    });
  }

  /* This onnxruntime-web build returns a plain object keyed by output name; Map-style access is
     kept so the page does not break on a runtime that returns one. */
  function out(result, name) {
    return result.get ? result.get(name) : result[name];
  }

  function infer() {
    var feed = { image: tensorFromPlate() };
    var base, evi;
    return SESS.baseline.run(feed).then(function (r) {
      base = r;
      return SESS.evidential.run(feed);
    }).then(function (r) {
      evi = r;
      var logits = Array.prototype.slice.call(out(base, "logits").data);
      var features = Array.prototype.slice.call(out(base, "features").data);
      var alpha = Array.prototype.slice.call(out(evi, "alpha").data);
      var p = Array.prototype.slice.call(out(evi, "p").data);
      var tau = out(evi, "tau").data[0];
      paint(reduceAll(logits, features, alpha, p, tau));
    });
  }

  /* ---------------------------------------------------------------- painting */

  var CLASSES = [];
  function bars(svg, probs, colour, marks) {
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    var rowH = 14, x0 = 62, w = 210, head = 22;
    text(svg, 0, 9, "class", "head");
    text(svg, x0, 9, "probability", "head");
    if (marks) text(svg, x0 + w + 52, 9, "evidence  α − 1", "head");
    CLASSES.forEach(function (name, i) {
      var y = i * rowH + head;
      text(svg, 0, y + 3, name.slice(0, 8), "rowlabel");
      line(svg, x0, y, x0 + w, y, "grid");
      line(svg, x0, y, x0 + Math.max(0.6, probs[i] * w), y, "series", colour, 1.6);
      text(svg, x0 + w + 6, y + 3, probs[i].toFixed(4), "value");
      if (marks && marks[i] !== undefined) {
        text(svg, x0 + w + 52, y + 3, marks[i], "value");
      }
    });
    svg.setAttribute("viewBox", "0 0 " + (marks ? 470 : 384) + " " +
      (CLASSES.length * rowH + head + 4));
  }
  function line(svg, x1, y1, x2, y2, cls, stroke, width) {
    var n = document.createElementNS("http://www.w3.org/2000/svg", "path");
    n.setAttribute("d", "M" + x1.toFixed(1) + " " + y1.toFixed(1) + "L" + x2.toFixed(1) + " " + y2.toFixed(1));
    if (cls) n.setAttribute("class", cls);
    if (stroke) n.setAttribute("stroke", stroke);
    if (width) n.setAttribute("stroke-width", width);
    svg.appendChild(n);
    return n;
  }
  function text(svg, x, y, str, cls) {
    var n = document.createElementNS("http://www.w3.org/2000/svg", "text");
    n.setAttribute("x", x); n.setAttribute("y", y);
    if (cls) n.setAttribute("class", cls);
    n.textContent = str;
    svg.appendChild(n);
    return n;
  }
  function readout(host, rows) {
    host.textContent = "";
    rows.forEach(function (r) {
      var d = document.createElement("div");
      var dt = document.createElement("dt"); dt.textContent = r[0];
      var dd = document.createElement("dd"); dd.textContent = r[1];
      d.appendChild(dt); d.appendChild(dd); host.appendChild(d);
    });
  }

  var HUE = { 486: "#0d6ba6", 656: "#ab2a1e", 589: "#7c600c" };

  function paint(all) {
    var a1 = all.A1_fedl, b1 = all.B1_ce, fd = all.A1_internals;
    bars(document.getElementById("bars-b1"), b1.mean, HUE[656]);
    readout(document.getElementById("read-b1"), [
      ["Predicted", CLASSES[b1.predicted]],
      ["Probability", b1.confidence.toFixed(5)],
      ["Epistemic", "not defined"],
      ["Aleatoric", b1.aleatoric.toFixed(5)]
    ]);

    bars(document.getElementById("bars-a1"), a1.mean, HUE[486],
      fd.alpha.map(function (v) { return Math.max(v - 1, 0).toFixed(1); }));
    readout(document.getElementById("read-a1"), [
      ["Predicted", CLASSES[a1.predicted]],
      ["Probability", a1.confidence.toFixed(5)],
      ["Epistemic", a1.epistemic.toFixed(5)],
      ["Aleatoric", a1.aleatoric.toFixed(5)],
      ["Total evidence", fd.alpha0.toFixed(1)],
      ["Tau", fd.tau.toFixed(3)]
    ]);

    var limit = INSTR.abstain.A1_fedl.total_uncertainty;
    var declining = a1.total > limit;
    var verdict = document.getElementById("verdict-a1");
    verdict.textContent = declining
      ? "Declining to answer: total uncertainty " + a1.total.toFixed(4) + " is past the line at " +
        limit.toFixed(4) + ", where the test risk-coverage curve passes 5 % risk."
      : "Answering: total uncertainty " + a1.total.toFixed(4) + " is inside the line at " +
        limit.toFixed(4) + ".";
    verdict.className = "instrument__verdict" + (declining ? " is-declining" : "");
    document.getElementById("inst-a1").setAttribute("data-state",
      declining ? "declining" : "answering");

    table(all);
  }

  var ORDER = ["B1_ce", "B2_ce_temperature", "A1_fedl", "A2_fedl_dirichlet", "A3_laplace_icla"];
  var SHORT = { B1_ce: "B1 · cross-entropy", B2_ce_temperature: "B2 · B1 + temperature",
                A1_fedl: "A1 · F-EDL evidential",
                A2_fedl_dirichlet: "A2 · A1 + Dirichlet scaling",
                A3_laplace_icla: "A3 · B1 + ICLA Laplace" };
  /* The codes are the report's shorthand; a visitor deserves to know what each one actually is. */
  var DEFINED = [
    ["B1", "cross-entropy baseline", "VGG-16-BN with spectral normalisation on every conv and linear layer, trained with ordinary cross-entropy. One weight vector, one softmax answer, no uncertainty term. The reference the two Bayesian models have to beat."],
    ["B2", "B1 + temperature scaling", "The same trained network, logits divided by one scalar T fitted on the validation split (Guo et al., 2017, On Calibration of Modern Neural Networks). The cheapest possible calibration: it changes confidence, not answers."],
    ["A1", "F-EDL, the algorithm under study", "Uncertainty Estimation by Flexible Evidential Deep Learning, Yoon, Kim et al., NeurIPS 2025, arXiv:2510.18322. Three heads on the same features emit alpha = exp(g1(z)), p = softmax(g2(z)) and tau = softplus(g3(z)), which together define a flexible Dirichlet over the ten classes. Evidence, mode and concentration are all read off this plate."],
    ["A2", "A1 + Dirichlet scaling", "A vector calibration in log-probability space, z' = t*log mu + b, with per-class t and b fitted on validation (Reea et al.; Kuhn et al.). Strictly wider than a single temperature, and it rescales probabilities only: the Dirichlet variance is not carried through, which is why its epistemic row is struck."],
    ["A3", "B1 + ICLA Laplace", "Identity Curvature Laplace Approximation for Improved Out-of-Distribution Detection, Zhdanov et al., WACV 2025, arXiv:2312.10464. A Gaussian posterior over the last layer's 5,120 weights, curvature replaced by the identity and prior precision lambda fitted by marginal likelihood. Uncertainty here comes from sampling weights, not from an evidence head."]
  ];
  var NO_EPISTEMIC = { B1_ce: 1, B2_ce_temperature: 1, A2_fedl_dirichlet: 1 };

  function table(all) {
    var host = document.getElementById("all-configs");
    var html = "<table><caption>The same specimen, five reductions</caption><thead><tr>" +
      "<th scope='col' class='row-head'>Configuration</th><th scope='col'>Predicted</th>" +
      "<th scope='col'>Probability</th><th scope='col'>Epistemic</th><th scope='col'>Aleatoric</th>" +
      "<th scope='col'>Declines</th></tr></thead><tbody>";
    ORDER.forEach(function (id) {
      var r = all[id];
      var limit = INSTR.abstain[id] ? INSTR.abstain[id].total_uncertainty : null;
      var declines = limit !== null && r.total > limit;
      html += "<tr><th scope='row' class='row-head'>" + SHORT[id] + "</th>" +
        "<td class='num'>" + CLASSES[r.predicted] + "</td>" +
        "<td class='num'>" + r.confidence.toFixed(4) + "</td>" +
        (NO_EPISTEMIC[id]
          ? "<td class='struck'><span class='strike' aria-hidden='true'></span>" +
            "<span class='strike-reason'>not defined</span></td>"
          : "<td class='num'>" + r.epistemic.toFixed(5) + "</td>") +
        "<td class='num'>" + r.aleatoric.toFixed(4) + "</td>" +
        "<td class='num'>" + (limit === null ? "—" : (declines ? "yes" : "no")) + "</td></tr>";
    });
    host.innerHTML = html + "</tbody></table>";
    var defs = document.createElement("dl");
    defs.className = "defs";
    DEFINED.forEach(function (d) {
      var dt = document.createElement("dt");
      dt.textContent = d[0] + " — " + d[1];
      var dd = document.createElement("dd");
      dd.textContent = d[2];
      defs.appendChild(dt);
      defs.appendChild(dd);
    });
    host.appendChild(defs);
  }

  /* ---------------------------------------------------------------- status */

  function setStatus(kind, html) {
    status.className = "bench__status is-" + kind;
    status.innerHTML = html;
  }

  /* ---------------------------------------------------------------- boot */

  function loadOrt() {
    var script = document.querySelector('script[src$="ort.all.min.js"]');
    ort.env.wasm.wasmPaths = new URL(".", script.src).href;
    ort.env.wasm.numThreads = 1;
  }

  function makeSession(buf) {
    var back = ["wasm"];
    return ort.InferenceSession.create(buf, { executionProviders: ["webgpu"].concat(back) })
      .then(function (s) { SESS.webgpu = true; return s; })
      .catch(function () {
        return ort.InferenceSession.create(buf, { executionProviders: ["wasm"] });
      });
  }

  function fetchModel(path) {
    return fetch(path).then(function (r) {
      if (!r.ok) throw new Error(path + " returned " + r.status);
      return r.arrayBuffer();
    });
  }

  function boot() {
    if (location.protocol === "file:") {
      setStatus("error", "Opened from disk, so the browser will not let this page read the two " +
        "models. Serve the folder: <code>python -m http.server 8801 --directory webapp</code> from " +
        "the project root, then open <code>http://127.0.0.1:8801/</code>. The " +
        "<a href='register.html'>measurement register</a> works from disk as it is.");
      return;
    }
    loadOrt();
    fetch("data/instruments.json").then(function (r) { return r.json(); })
    .then(function (instruments) {
      INSTR = instruments;
      CLASSES = INSTR.classes;
      document.getElementById("field-decline").textContent =
        "total uncertainty " + INSTR.abstain.A1_fedl.total_uncertainty.toFixed(4) +
        " (5 % risk line)";
      return Promise.all([fetchModel("model/baseline.onnx"), fetchModel("model/evidential.onnx")]);
    }).then(function (bufs) {
      setStatus("loading", "Networks downloaded; compiling for this machine. The first inference " +
        "is slower than the rest.");
      return Promise.all([makeSession(bufs[0]), makeSession(bufs[1])]);
    }).then(function (sessions) {
      SESS.baseline = sessions[0];
      SESS.evidential = sessions[1];
      buildWall();
      runtimeField.textContent = sessions[0].inputNames ? "onnxruntime-web, " +
        (SESS.webgpu ? "WebGPU" : "WASM (CPU)") : "onnxruntime-web";
      setStatus("ready", "Ready. Draw on the plate, or pick a reference specimen.");
      run();
    }).catch(function (err) {
      setStatus("error", "The instruments did not load: " + (err.message || err) + ". Check that " +
        "<code>model/baseline.onnx</code> and <code>model/evidential.onnx</code> exist — they are " +
        "written by <code>python src/export_onnx.py</code>.");
    });
  }


  document.querySelectorAll("[data-brush]").forEach(function (b) {
    b.addEventListener("click", function () {
      brush = parseInt(b.getAttribute("data-brush"), 10);
      erasing = false;
      document.querySelectorAll("[data-brush]").forEach(function (o) {
        o.setAttribute("aria-pressed", o === b ? "true" : "false");
      });
      document.getElementById("tool-eraser").setAttribute("aria-pressed", "false");
    });
  });
  document.getElementById("tool-eraser").addEventListener("click", function () {
    erasing = !erasing;
    this.setAttribute("aria-pressed", erasing ? "true" : "false");
  });
  document.getElementById("act-clear").addEventListener("click", function () {
    clearPlate(true);
    specimenField.textContent = "blank plate";
    run();
  });
  document.getElementById("act-noise").addEventListener("click", function () {
    var d = ctx.getImageData(0, 0, 32, 32);
    for (var i = 0; i < d.data.length; i += 4) {
      var v = Math.floor(Math.random() * 256);
      d.data[i] = v; d.data[i + 1] = Math.floor(Math.random() * 256);
      d.data[i + 2] = Math.floor(Math.random() * 256); d.data[i + 3] = 255;
    }
    ctx.putImageData(d, 0, 0);
    specimenField.textContent = "uniform noise";
    run();
  });
  document.getElementById("act-photo").addEventListener("click", function () {
    document.getElementById("input-photo").click();
  });
  document.getElementById("input-photo").addEventListener("change", function (e) {
    var file = e.target.files && e.target.files[0];
    if (!file) return;
    var url = URL.createObjectURL(file);
    var img = new Image();
    img.onload = function () {
      imageToPlate(img);
      URL.revokeObjectURL(url);
      specimenField.textContent = file.name;
      run();
    };
    img.src = url;
  });


  /* ---------------------------------------------------------------- the wall */

  var WALL = { idx: null, order: [], shown: 0, cls: "all", outcome: "all",
               key: "index", model: "A1_fedl", dir: 1 };
  var CHUNK = 60;
  var SORTS = [["index", "test index"], ["epistemic", "epistemic"],
               ["total", "total uncertainty"], ["aleatoric", "aleatoric"],
               ["confidence", "confidence"]];

  function chip(host, label, onPick, active) {
    var b = document.createElement("button");
    b.type = "button";
    b.className = "pick";
    b.textContent = label;
    b.setAttribute("aria-pressed", active ? "true" : "false");
    b.addEventListener("click", function () {
      Array.prototype.forEach.call(host.children, function (o) {
        o.setAttribute("aria-pressed", o === b ? "true" : "false");
      });
      onPick();
    });
    host.appendChild(b);
    return b;
  }

  function wallValue(rec, key, model) {
    var m = rec.m[model];
    if (key === "index") return rec.i;
    if (key === "predicted") return m[0];
    if (key === "confidence") return m[1];
    if (key === "epistemic") return m[2];
    if (key === "aleatoric") return m[3];
    return m[4];
  }

  function resort() {
    var recs = WALL.idx.records;
    var limit = INSTR.abstain.A1_fedl.total_uncertainty;
    var keep = [];
    for (var i = 0; i < recs.length; i++) {
      var r = recs[i];
      if (WALL.cls !== "all" && r.c !== WALL.cls) continue;
      if (WALL.outcome === "wrong" && r.m.B1_ce[0] === r.y) continue;
      if (WALL.outcome === "right" && r.m.B1_ce[0] !== r.y) continue;
      if (WALL.outcome === "declined" && r.m.A1_fedl[4] <= limit) continue;
      keep.push(r);
    }
    if (WALL.key !== "index") {
      keep.sort(function (a, b) {
        return (wallValue(b, WALL.key, WALL.model) - wallValue(a, WALL.key, WALL.model)) *
          (WALL.dir === -1 ? 1 : -1);
      });
    } else {
      keep.sort(function (a, b) { return (a.i - b.i) * (WALL.dir === -1 ? -1 : 1); });
    }
    WALL.order = keep;
    WALL.shown = 0;
    var grid = document.getElementById("pick-grid");
    grid.textContent = "";
    count();
    more();
  }

  function count() {
    var sortName = "test index";
    SORTS.forEach(function (k) { if (k[0] === WALL.key) sortName = k[1]; });
    document.getElementById("pick-count").textContent =
      WALL.order.length.toLocaleString("en-GB") + " of " +
      WALL.idx.count.toLocaleString("en-GB") + " images · " +
      (WALL.key === "index"
        ? "in test order"
        : "sorted by " + WALL.model.split("_")[0] + " " + sortName + ", " +
          (WALL.dir === -1 ? "high to low" : "low to high")) +
      " \u00b7 showing the first " + WALL.shown.toLocaleString("en-GB");
  }

  function more() {
    var grid = document.getElementById("pick-grid");
    var stop = Math.min(WALL.order.length, WALL.shown + CHUNK);
    var max = 1e-12, min = Infinity;
    if (WALL.key !== "index") {
      WALL.order.forEach(function (r) {
        var v = wallValue(r, WALL.key, WALL.model);
        if (v > max) max = v;
        if (v < min) min = v;
      });
    }
    for (var i = WALL.shown; i < stop; i++) {
      var r = WALL.order[i];
      var b = document.createElement("button");
      b.type = "button";
      var img = document.createElement("img");
      img.src = "data/test/" + String(r.i).padStart(5, "0") + ".png";
      img.alt = "Test image " + r.i + ", true class " + r.c;
      img.width = 32; img.height = 32; img.loading = "lazy";
      b.appendChild(img);
      var v = wallValue(r, WALL.key, WALL.model);
      b.style.setProperty("--u", WALL.key === "index" ? 0
        : Math.max(0.02, (v - min) / (max - min)).toFixed(3));
      b.style.color = HUE[486];
      if (r.m.B1_ce[0] !== r.y) b.classList.add("miss");
      b.title = "#" + r.i + " \u00b7 true " + r.c +
        " \u00b7 B1 says " + WALL.idx.classes[r.m.B1_ce[0]] + " at " + r.m.B1_ce[1].toFixed(4) +
        " \u00b7 A1 epistemic " + r.m.A1_fedl[2].toFixed(6) +
        " \u00b7 A1 total " + r.m.A1_fedl[4].toFixed(5);
      (function (rec) {
        b.addEventListener("click", function () {
          var probe = new Image();
          probe.onload = function () {
            imageToPlate(probe);
            specimenField.textContent = "test_" + String(rec.i).padStart(5, "0") +
              " \u00b7 true class " + rec.c;
            run();
            document.querySelector(".bench__grid").scrollIntoView({ block: "start",
              behavior: scrollBehaviour() });
          };
          probe.src = "data/test/" + String(rec.i).padStart(5, "0") + ".png";
        });
      })(r);
      grid.appendChild(b);
    }
    WALL.shown = stop;
    document.getElementById("pick-more").textContent = WALL.shown < WALL.order.length
      ? "scroll for " + (WALL.order.length - WALL.shown).toLocaleString("en-GB") + " more"
      : "end of selection";
    count();
  }

  function buildWall() {
    fetch("data/test-index.json").then(function (r) {
      if (!r.ok) throw new Error("test-index.json returned " + r.status);
      return r.json();
    }).then(function (idx) {
      WALL.idx = idx;
      var clsHost = document.getElementById("pick-classes");
      chip(clsHost, "all", function () { WALL.cls = "all"; resort(); }, true);
      idx.classes.forEach(function (c) {
        chip(clsHost, c, function () { WALL.cls = c; resort(); }, false);
      });
      var outHost = document.getElementById("pick-outcomes");
      chip(outHost, "everything", function () { WALL.outcome = "all"; resort(); }, true);
      chip(outHost, "baseline wrong", function () { WALL.outcome = "wrong"; resort(); }, false);
      chip(outHost, "baseline right", function () { WALL.outcome = "right"; resort(); }, false);
      chip(outHost, "declined by A1", function () { WALL.outcome = "declined"; resort(); }, false);

      var keyHost = document.getElementById("pick-sortkeys");
      SORTS.forEach(function (k) {
        chip(keyHost, k[1], function () {
          if (WALL.key === k[0]) WALL.dir = -WALL.dir; else { WALL.key = k[0]; WALL.dir = -1; }
          resort();
        }, k[0] === WALL.key);
      });
      var modelHost = document.getElementById("pick-sortmodels");
      ORDER.forEach(function (id) {
        chip(modelHost, id.split("_")[0], function () { WALL.model = id; resort(); },
          id === WALL.model);
      });

      if ("IntersectionObserver" in window) {
        new IntersectionObserver(function (entries) {
          if (entries[0].isIntersecting && WALL.shown < WALL.order.length) more();
        }, { root: document.getElementById("pick-grid"), rootMargin: "200px" })
          .observe(document.getElementById("pick-grid"));
      }
      resort();
    }).catch(function (err) {
      document.getElementById("pick-count").textContent =
        "The whole-station index is not available: " + (err.message || err) +
        ". Regenerate it with python src/export_test_wall.py. The plate above still works.";
    });
  }

  boot();
})();
