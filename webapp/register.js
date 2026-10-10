/* CIFAR-10 observation register — the interface required by assignment component (6).
 *
 * Project : CISC3024 AI Assignment #2 (Pattern Recognition, University of Macau)
 * Author  : AI-generated implementation (Qoder). The student wrote no code.
 *
 * Plain browser code: no build step, no framework, no chart library. Every chart is hand-authored
 * SVG, and every value is read from data/dataset.js, which src/export_webapp_data.py generates from
 * the same result files the report tables use. Nothing here recomputes a metric.
 *
 * Register grammar: colour identifies a series (one of seven calibrated wavelengths); LINE FORM
 * carries state — solid = measured, dashed = ablation, half-height = partial, struck = the quantity
 * does not exist for that configuration and is never shown as zero. */

(function () {
  "use strict";

  var DATA = window.BAYES_DATA;
  var entries = document.getElementById("entries");
  var railBox = document.getElementById("rail-ticks");
  var colophon = document.getElementById("colophon");

  if (!DATA || !DATA.manifest || !DATA.samples) {
    entries.innerHTML = '<p class="degraded">The register loaded but found no reduction data. ' +
      '<code>data/dataset.js</code> is missing or empty. Regenerate it from the repository root with ' +
      '<code>python src/export_webapp_data.py --derived-only</code>, then reload this page.</p>';
    return;
  }

  var M = DATA.manifest;
  var S = DATA.samples;

  /* ------------------------------------------------------------ series */

  var SERIES = [
    { id: "B1_ce", short: "B1", name: "cross-entropy baseline", wl: 656, epistemic: false,
      why: "a softmax point estimate: no weight or sample uncertainty is carried" },
    { id: "B2_ce_temperature", short: "B2", name: "B1 + temperature scaling", wl: 615, epistemic: false,
      why: "temperature rescales logits only; the variance term stays undefined" },
    { id: "A1_fedl", short: "A1", name: "flexible evidential (F-EDL)", wl: 486, epistemic: true,
      why: "" },
    { id: "A2_fedl_dirichlet", short: "A2", name: "A1 + Dirichlet scaling", wl: 546, epistemic: false,
      why: "the rescaling fits probabilities only; Dirichlet variance is not carried through it" },
    { id: "A3_laplace_icla", short: "A3", name: "B1 + ICLA Laplace", wl: 436, epistemic: true,
      why: "" }
  ];
  var REF = 405;    /* reference marks: the ideal, chance, the closed form */
  var CURSOR = 589; /* whatever the reader has selected */

  var INK = { 405: "#4a3598", 436: "#1c4fbe", 486: "#0d6ba6", 546: "#1b7a4b",
              589: "#7c600c", 615: "#a8551a", 656: "#ab2a1e" };

  var byId = {};
  SERIES.forEach(function (s) { byId[s.id] = s; });
  var IDS = SERIES.map(function (s) { return s.id; });

  /* ------------------------------------------------------------ metrics */

  var OOD_CONTROLS = { near_svhn: "SVHN (near)", far_tiny_imagenet: "Tiny-ImageNet (far)" };
  var OOD_SCORES = { total: "total", aleatoric: "aleatoric", epistemic: "epistemic" };
  var OOD_SHORT = { near_svhn: "SVHN", far_tiny_imagenet: "Tiny-ImageNet" };
  var LABELS = {
    top1: "Top-1 accuracy", top2: "Top-2 accuracy", macro_f1: "Macro F1",
    nll: "Negative log-likelihood", brier: "Brier score",
    ece: "Expected calibration error", adaptive_ece: "Adaptive ECE",
    classwise_ece: "Classwise ECE", aurc: "AURC (risk-coverage area)",
    cov_at_risk5: "Coverage at 5% risk", risk_at_cov80: "Risk at 80% coverage",
    err_error_auroc: "Error detection AUROC", err_error_aupr: "Error detection AUPR",
    ale_error_auroc: "Error AUROC, aleatoric score", ale_error_aupr: "Error AUPR, aleatoric score",
    epistemic_mean: "Mean epistemic uncertainty", aleatoric_mean: "Mean aleatoric uncertainty",
    mean_confidence_minus_accuracy: "Confidence minus accuracy gap"
  };
  var HIGH_BETTER = { top1: 1, top2: 1, macro_f1: 1, cov_at_risk5: 1 };
  var DP = { epistemic_mean: 5 };

  function labelOf(key) {
    if (LABELS[key]) return LABELS[key];
    var m = key.match(/^(near_svhn|far_tiny_imagenet)_(total|aleatoric|epistemic)_(auroc|aupr|fpr95)$/);
    if (!m) return key;
    return OOD_SHORT[m[1]] + " · " + OOD_SCORES[m[2]] + " " + m[3].toUpperCase();
  }
  function higherBetter(key) {
    if (HIGH_BETTER[key]) return true;
    if (/_fpr95$/.test(key)) return false;
    return /_(auroc|aupr)$/.test(key);
  }
  function dpOf(key) { return DP[key] || 4; }
  function fmt(v, dp) {
    if (v === null || v === undefined || !isFinite(v)) return "—";
    return (v < 0 ? "−" : "") + Math.abs(v).toFixed(dp);
  }
  function stats(modelId, key, src) {
    var table_ = src || M.models;
    var e = table_[modelId] && table_[modelId][key];
    return e && typeof e.mean === "number" ? e : null;
  }
  /* Struck = the configuration does not define the quantity, even though the reduction formula
     returns a number for it (0, or the degenerate 0.5 of a constant score). */
  function struckReason(modelId, key) {
    var s = byId[modelId];
    if (!s || s.epistemic) return "";
    if (key === "epistemic_mean" || key === "err_error_auroc" || key === "err_error_aupr" ||
        /_epistemic_(auroc|aupr|fpr95)$/.test(key)) return s.why;
    return "";
  }

  /* ------------------------------------------------------------ svg */

  var NS = "http://www.w3.org/2000/svg";
  function node(tag, attrs, parent) {
    var n = document.createElementNS(NS, tag);
    for (var k in attrs) {
      if (attrs[k] === null || attrs[k] === undefined) continue;
      if (k === "text") { n.textContent = attrs[k]; continue; }
      n.setAttribute(k, attrs[k]);
    }
    if (parent) parent.appendChild(n);
    return n;
  }
  function line(d, cls, colour, parent, extra) {
    var a = { d: d, class: cls || null, fill: "none" };
    if (colour) a.stroke = colour;
    for (var k in (extra || {})) a[k] = extra[k];
    return node("path", a, parent);
  }
  /* A plate is one labelled drawing: box gives the plot rectangle and both domains. */
  function plate(host, w, h, box, aria) {
    /* The width and height attributes give the drawing an intrinsic size, which is what lets a
       narrow screen fold it out at full scale instead of shrinking its labels below reading size. */
    var svg = node("svg", { viewBox: "0 0 " + w + " " + h, width: w, height: h,
      role: "img", "aria-label": aria || "" }, host);
    svg._b = box;
    svg._X = function (v) { return box.x0 + (v - box.lo) / (box.hi - box.lo) * (box.x1 - box.x0); };
    svg._Y = function (v) { return box.y1 + (v - box.ylo) / (box.yhi - box.ylo) * (box.y0 - box.y1); };
    svg._w = w; svg._h = h;
    return svg;
  }
  function axes(svg, xs, ys, dp) {
    var b = svg._b;
    line("M" + b.x0 + " " + b.y0 + "V" + b.y1, "axis", null, svg);
    line("M" + b.x0 + " " + b.y1 + "H" + b.x1, "axis", null, svg);
    (xs || []).forEach(function (t) {
      var x = svg._X(t);
      line("M" + x + " " + b.y1 + "v3", "grid", null, svg);
      node("text", { x: x, y: b.y1 + 13, "text-anchor": "middle", text: fmt(t, dp) }, svg);
    });
    (ys || []).forEach(function (t) {
      var y = svg._Y(t);
      line("M" + (b.x0 - 3) + " " + y + "H" + b.x1, "grid", null, svg);
      node("text", { x: b.x0 - 6, y: y + 3, "text-anchor": "end", text: fmt(t, dp) }, svg);
    });
  }
  /* Rows read top to bottom, the way a register page is ruled. */
  function rowY(svg, i, n) {
    var b = svg._b;
    return b.y0 + (i + 0.5) * (b.y1 - b.y0) / n;
  }
  function struck(svg, x, y, len) {
    return line("M" + x + " " + y + "h" + len + "M" + (x + 2) + " " + (y + 4) + "l" +
                (len - 4) + " -8", "state-struck", null, svg);
  }

  /* ------------------------------------------------------------ shared blocks */

  function note(host, html) {
    var p = document.createElement("p");
    p.className = "note";
    p.innerHTML = html;
    host.appendChild(p);
  }
  function small(host, html) {
    var p = document.createElement("p");
    p.className = "small";
    p.innerHTML = html;
    host.appendChild(p);
  }
  function sub(host, text) {
    var h = document.createElement("h3");
    h.className = "sub";
    h.textContent = text;
    host.appendChild(h);
  }
  function legend(host) {
    var box = document.createElement("div");
    box.className = "state-legend";
    box.innerHTML = "<span><i></i>measured</span><span><i class=\"dashed\"></i>ablation</span>" +
      "<span><i class=\"half\"></i>partial run</span><span><i class=\"struck\"></i>not defined</span>" +
      "<span>Colour marks the series; the rule itself marks its state.</span>";
    host.appendChild(box);
  }
  function seriesLabel(s) {
    var key = document.createElement("span");
    key.className = "series-key";
    var sw = document.createElement("span");
    sw.className = "series-key__rule";
    sw.style.color = INK[s.wl];
    var t = document.createElement("span");
    t.textContent = s.short + " · " + s.name;
    key.appendChild(sw);
    key.appendChild(t);
    return key;
  }
  function controlBar(host, labelText, items, onChange, current) {
    var bar = document.createElement("div");
    bar.className = "controls";
    var lab = document.createElement("span");
    lab.className = "controls__label";
    lab.textContent = labelText;
    bar.appendChild(lab);
    items.forEach(function (it) {
      var b = document.createElement("button");
      b.type = "button";
      b.className = "pick";
      b.textContent = it.label;
      b.setAttribute("aria-pressed", it.value === current ? "true" : "false");
      b.addEventListener("click", function () {
        bar.querySelectorAll(".pick").forEach(function (o) {
          o.setAttribute("aria-pressed", o === b ? "true" : "false");
        });
        onChange(it.value);
      });
      bar.appendChild(b);
    });
    host.appendChild(bar);
  }

  /* ------------------------------------------------------------ table cells */

  function whisker(std, scaleMax) {
    var w = 34, half = w / 2;
    var len = Math.max(1.5, Math.min(half - 1, (std / scaleMax) * (half - 1)));
    var svg = node("svg", { class: "whisker", viewBox: "0 0 " + w + " 9", width: w, height: 9,
      "aria-hidden": "true" });
    node("path", { d: "M" + (half - len) + " 4.5h" + (len * 2) + "M" + (half - len) + " 2v5M" +
      (half + len) + " 2v5M" + half + " 3.4v2.2" }, svg);
    return svg;
  }
  function struckCell(key) {
    var td = document.createElement("td");
    td.className = "struck";
    var bar = document.createElement("span");
    bar.className = "strike";
    bar.setAttribute("aria-hidden", "true");
    td.appendChild(bar);
    var why = document.createElement("span");
    why.className = "strike-reason";
    why.textContent = "not defined";
    td.appendChild(why);
    var sr = document.createElement("span");
    sr.className = "sr";
    sr.textContent = "not defined for this configuration";
    td.appendChild(sr);
    return td;
  }
  function valueCell(modelId, key, src, scaleMax, best) {
    var reason = struckReason(modelId, key);
    var td = document.createElement("td");
    if (reason) {
      td.className = "struck";
      td.title = "Not defined: " + reason;
      var bar = document.createElement("span");
      bar.className = "strike";
      bar.setAttribute("aria-hidden", "true");
      td.appendChild(bar);
      var why = document.createElement("span");
      why.className = "strike-reason";
      why.textContent = "not defined";
      td.appendChild(why);
      var sr = document.createElement("span");
      sr.className = "sr";
      sr.textContent = "not defined: " + reason;
      td.appendChild(sr);
      return td;
    }
    var st = stats(modelId, key, src);
    if (!st) { td.textContent = "—"; return td; }
    if (best) td.className = "is-best";
    var wrap = document.createElement("span");
    wrap.className = "cellv";
    var num = document.createElement("span");
    num.className = "num";
    num.textContent = fmt(st.mean, dpOf(key));
    wrap.appendChild(num);
    if (scaleMax) wrap.appendChild(whisker(st.std, scaleMax));
    td.appendChild(wrap);
    var alt = document.createElement("span");
    alt.className = "sr";
    alt.textContent = " plus or minus " + fmt(st.std, dpOf(key)) + " over " + st.n + " seeds";
    td.appendChild(alt);
    return td;
  }
  function table(host, opts) {
    var rows = opts.rows, keys = opts.keys, src = opts.src || M.models;
    var scales = {}, best = {};
    keys.forEach(function (k) {
      var mx = 0, bestMean = null, hi = higherBetter(k);
      rows.forEach(function (r) {
        var st = stats(r, k, src);
        if (!st || struckReason(r, k)) return;
        if (st.std > mx) mx = st.std;
        if (bestMean === null || (hi ? st.mean > bestMean : st.mean < bestMean)) bestMean = st.mean;
      });
      scales[k] = mx || 1;
      best[k] = bestMean;
    });
    var wrap = document.createElement("div");
    wrap.className = "table-wrap" + (opts.dashed ? " state-dashed" : "");
    var t = document.createElement("table");
    if (opts.caption) {
      var cap = document.createElement("caption");
      cap.textContent = opts.caption;
      t.appendChild(cap);
    }
    var thead = node_remove(document.createElement("thead"), t);
    var hr = document.createElement("tr");
    var th0 = document.createElement("th");
    th0.setAttribute("scope", "col");
    th0.className = "row-head";
    th0.textContent = opts.rowLabel || "Configuration";
    hr.appendChild(th0);
    keys.forEach(function (k) {
      var th = document.createElement("th");
      th.setAttribute("scope", "col");
      th.textContent = labelOf(k);
      th.title = (higherBetter(k) ? "higher is better" : "lower is better");
      hr.appendChild(th);
    });
    thead.appendChild(hr);
    var tb = node_remove(document.createElement("tbody"), t);
    rows.forEach(function (r) {
      var tr = document.createElement("tr");
      var name = document.createElement("th");
      name.setAttribute("scope", "row");
      name.className = "row-head";
      name.appendChild(seriesLabel(byId[r]));
      tr.appendChild(name);
      keys.forEach(function (k) {
        var st = stats(r, k, src);
        tr.appendChild(valueCell(r, k, src, scales[k],
          st && best[k] !== null && st.mean === best[k]));
      });
      tb.appendChild(tr);
    });
    wrap.appendChild(t);
    host.appendChild(wrap);
    return wrap;
  }
  function node_remove(el, parent) { parent.appendChild(el); return el; }
  function perSeed(seed) {
    for (var i = 0; i < M.per_seed.length; i++) {
      if (M.per_seed[i].seed === seed) return M.per_seed[i];
    }
    return { models: {}, laplace: null, calibrators: {} };
  }
  function seedValue(seed, modelId, key) {
    var m = perSeed(seed).models[modelId];
    return m && typeof m[key] === "number" ? m[key] : null;
  }

  /* ============================================================ I reduction */
  var FILL = {};

  FILL.headline = function (host) {
    legend(host);
    note(host, "Mean over three seeds of the same training and evaluation code path, on the frozen test " +
      "split. A whisker is one standard deviation drawn against that column's largest spread; the prime " +
      "marks the best value in a column among the configurations that define it. A struck cell is a " +
      "quantity the configuration does not define &mdash; not a zero, and not the 0.5 a ranking formula " +
      "returns when its score is constant.");
    table(host, {
      rows: IDS,
      keys: ["top1", "top2", "nll", "brier", "ece", "aurc", "err_error_auroc", "epistemic_mean"],
      caption: "Headline reduction · CIFAR-10 test split · 10 000 images"
    });
    sub(host, "Complete reduction");
    note(host, "Every quantity the evaluation recorded, in the order the reduction code writes it. " +
      "Nothing here was re-entered by hand: this table reads the same JSON as the one above.");
    var all = Object.keys(M.models.B1_ce);
    table(host, { rows: IDS, keys: all,
      caption: "All " + all.length + " recorded quantities · three seeds" });
    small(host, "Abbreviations: ECE is expected calibration error in equal-width confidence bins, AUROC " +
      "is a score's ranking against the binary event it should separate, FPR95 is the false-positive " +
      "rate at 95% true-positive rate, and AURC is the area under the risk-coverage curve in entry III.");
  };

  /* ============================================================ II reliability */
  FILL.reliability = function (host) {
    note(host, "Fifteen equal-width bins over mean predicted probability. Each rule sits at the accuracy " +
      "that bin actually delivered, above or below its own confidence; the dotted line is perfect " +
      "reliability and the pale connector is the deviation. Bins that received no images are left blank " +
      "rather than drawn at zero.");
    var grid = document.createElement("div");
    grid.className = "plates";
    host.appendChild(grid);
    SERIES.forEach(function (s) {
      var fig = document.createElement("figure");
      grid.appendChild(fig);
      var w = 248, h = 208;
      var svg = plate(fig, w, h, { x0: 32, x1: w - 10, y0: 12, y1: h - 34,
        lo: 0, hi: 1, ylo: 0, yhi: 1 }, "Reliability diagram, " + s.name + ", seed " +
        M.experiment.seeds[0]);
      svg.setAttribute("class", "state-partial");
      axes(svg, [0, 0.5, 1], [0, 0.5, 1], 1);
      line("M" + svg._X(0) + " " + svg._Y(0) + "L" + svg._X(1) + " " + svg._Y(1), "ref", INK[REF], svg);
      var c = M.curves[s.id];
      if (!c) return;
      var maxCount = Math.max.apply(null, c.reliability_bins.map(function (b) { return b.count; }));
      c.reliability_bins.forEach(function (b) {
        if (!b.count || b.confidence === null) {
          line("M" + svg._X(b.bin_lo) + " " + svg._Y(0) + "v-3", "grid", null, svg);
          return;
        }
        var y = svg._Y(b.accuracy);
        var rule = line("M" + svg._X(b.bin_lo) + " " + y + "H" + svg._X(b.bin_hi), "series",
          INK[s.wl], svg, { "stroke-width": 1.3 });
        node("title", { text: "confidence " + fmt(b.confidence, 3) + ", accuracy " +
          fmt(b.accuracy, 3) + ", n " + b.count }, rule);
        if (Math.abs(b.accuracy - b.confidence) > 0.004) {
          line("M" + (svg._X(b.bin_lo) + svg._X(b.bin_hi)) / 2 + " " + y + "V" +
            svg._Y(b.confidence), "grid", null, svg);
        }
        line("M" + (svg._X(b.bin_lo) + svg._X(b.bin_hi)) / 2 + " " + svg._Y(0) + "V" +
          (svg._Y(0) - 13 * b.count / maxCount), "count", INK[s.wl], svg,
          { "stroke-opacity": 0.5 });
      });
      var cap = document.createElement("figcaption");
      cap.innerHTML = "<span>" + s.short + "</span><span>ECE <span class=\"num\">" + fmt(c.ece, 4) +
        "</span> · ACE <span class=\"num\">" + fmt(c.adaptive_ece, 4) + "</span></span>";
      fig.appendChild(cap);
    });
    small(host, "A short stub below each rule is that bin's image count, drawn against the plate's " +
      "largest bin; hover a rule for the exact confidence, accuracy and count.");
  };

  /* ============================================================ III risk-coverage */
  FILL.risk = function (host) {
    note(host, "Refuse the least confident images first: coverage is the fraction retained, risk the " +
      "error rate among them. Each curve ends at its own test error, between .0882 and .0928, so the " +
      "reading is how quickly each one falls and where it flattens, not where it starts.");
    var maxRisk = 0;
    SERIES.forEach(function (s) {
      var c = M.curves[s.id];
      if (c) c.risk_coverage.risk.forEach(function (v) { if (v > maxRisk) maxRisk = v; });
    });
    var hi = Math.ceil(maxRisk * 50) / 50;
    var fig = document.createElement("figure");
    host.appendChild(fig);
    var w = 660, h = 236;
    var svg = plate(fig, w, h, { x0: 48, x1: w - 152, y0: 14, y1: h - 34,
      lo: 0, hi: 1, ylo: 0, yhi: hi }, "Risk against coverage, five configurations, seed " +
      M.experiment.seeds[0]);
    svg.setAttribute("class", "state-partial");
    var ys = [];
    for (var g = 0; g <= hi + 1e-9; g += 0.02) ys.push(Number(g.toFixed(2)));
    axes(svg, [0, 0.25, 0.5, 0.75, 1], ys, 2);
    var ends = [];
    SERIES.forEach(function (s) {
      var c = M.curves[s.id];
      if (!c) return;
      var cov = c.risk_coverage.coverage, risk = c.risk_coverage.risk, d = "";
      for (var i = 0; i < cov.length; i++) {
        d += (i ? "L" : "M") + svg._X(cov[i]).toFixed(2) + " " + svg._Y(risk[i]).toFixed(2);
      }
      line(d, "series", INK[s.wl], svg);
      var aurc = stats(s.id, "aurc");
      node("title", { text: s.name + " · AURC " + (aurc ? fmt(aurc.mean, 4) : "—") }, svg.lastChild);
      ends.push({ s: s, y: svg._Y(risk[risk.length - 1]), aurc: aurc });
    });
    /* The five curves leave the frame within a few thousandths of each other, so their labels are
       spaced apart and joined by a leader rather than stacked where they would collide. */
    ends.sort(function (a, b) { return a.y - b.y; });
    var slot = 13, top = Math.min.apply(null, ends.map(function (e) { return e.y; }));
    ends.forEach(function (e, i) {
      var ly = top + i * slot;
      line("M" + (svg._X(1) + 2) + " " + e.y.toFixed(1) + "H" + (svg._X(1) + 9) + "V" +
        (ly - 3).toFixed(1), "grid", null, svg);
      node("text", { x: svg._X(1) + 13, y: ly, fill: INK[e.s.wl], class: "value",
        text: e.s.short + "  AURC " + (e.aurc ? fmt(e.aurc.mean, 4) : "—") }, svg);
    });
    var cap = document.createElement("figcaption");
    cap.innerHTML = "<span>Horizontal: coverage</span><span>Vertical: risk, 0 to " + fmt(hi, 2) +
      "</span>";
    fig.appendChild(cap);
    table(host, { rows: IDS, keys: ["aurc", "cov_at_risk5", "risk_at_cov80",
      "mean_confidence_minus_accuracy"], caption: "Selective prediction · three seeds" });
  };

  /* ============================================================ IV OOD */
  FILL.ood = function (host) {
    legend(host);
    note(host, "Two negative controls that never entered training, scored by the same posterior. AUROC " +
      "of 0.50 is chance, so a constant epistemic score is drawn struck rather than as the 0.5 its " +
      "formula returns &mdash; the degenerate value and the missing quantity look identical in a table " +
      "and mean different things.");
    Object.keys(OOD_CONTROLS).forEach(function (ctrl) {
      sub(host, OOD_CONTROLS[ctrl]);
      var fig = document.createElement("figure");
      host.appendChild(fig);
      var w = 660, h = 168;
      var svg = plate(fig, w, h, { x0: 52, x1: w - 20, y0: 16, y1: h - 30, lo: 0.5, hi: 1,
        ylo: 0, yhi: SERIES.length }, "AUROC against " + OOD_CONTROLS[ctrl]);
      line("M" + svg._X(0.5) + " " + svg._b.y0 + "V" + svg._b.y1, "ref", INK[REF], svg);
      [0.5, 0.6, 0.7, 0.8, 0.9, 1].forEach(function (t) {
        line("M" + svg._X(t) + " " + svg._b.y1 + "v4", "grid", null, svg);
        node("text", { x: svg._X(t), y: h - 8, "text-anchor": "middle", text: t.toFixed(1) }, svg);
      });
      SERIES.forEach(function (s, i) {
        var y = rowY(svg, i, SERIES.length);
        node("text", { x: 0, y: y + 3, class: "value", text: s.short }, svg);
        line("M" + 34 + " " + y + "H" + svg._X(1), "grid", null, svg);
        [["total", 6], ["epistemic", -6]].forEach(function (pair) {
          var key = ctrl + "_" + pair[0] + "_auroc";
          var st = stats(s.id, key);
          if (!st) return;
          if (struckReason(s.id, key)) {
            struck(svg, svg._X(0.5) + 6, y + pair[1], 30);
            node("text", { x: svg._X(0.5) + 42, y: y + pair[1] + 3,
              text: pair[0] + " not defined" }, svg);
            return;
          }
          var x = svg._X(Math.max(0.5, Math.min(1, st.mean)));
          line("M" + x + " " + (y + pair[1] - 5) + "V" + (y + pair[1] + 5), "tickmark", INK[s.wl], svg);
          node("title", { text: s.name + " · " + pair[0] + " AUROC " + fmt(st.mean, 4) + " ± " +
            fmt(st.std, 4) }, svg.lastChild);
          node("text", { x: x + 5, y: y + pair[1] + 3, fill: INK[s.wl],
            text: fmt(st.mean, 3) }, svg);
        });
      });
      var cap = document.createElement("figcaption");
      cap.innerHTML = "<span>Below the row: total uncertainty · above the row: epistemic</span>" +
        "<span>Chance 0.50</span>";
      fig.appendChild(cap);
      table(host, { rows: IDS, caption: "Scored against " + OOD_CONTROLS[ctrl],
        keys: [ctrl + "_total_auroc", ctrl + "_total_fpr95", ctrl + "_aleatoric_auroc",
               ctrl + "_epistemic_auroc", ctrl + "_epistemic_fpr95"] });
    });
  };

  /* ============================================================ V decomposition */
  FILL.decomposition = function (host) {
    legend(host);
    note(host, "Predictive uncertainty separates into an <strong>epistemic</strong> term more data " +
      "could reduce and an <strong>aleatoric</strong> term it cannot. Three of the five configurations " +
      "have no epistemic term at all, so those rules are struck: showing them as zero would let a " +
      "missing mechanism read as a measured advantage.");
    var maxAle = 0;
    SERIES.forEach(function (s) {
      var st = stats(s.id, "aleatoric_mean");
      if (st && st.mean > maxAle) maxAle = st.mean;
    });
    var hi = Math.ceil(maxAle * 25) / 25 + 0.02;
    var fig = document.createElement("figure");
    host.appendChild(fig);
    var w = 660, h = 200;
    var svg = plate(fig, w, h, { x0: 52, x1: w - 132, y0: 18, y1: h - 34, lo: 0, hi: hi,
      ylo: 0, yhi: SERIES.length }, "Epistemic and aleatoric uncertainty per configuration");
    [0, hi / 2, hi].forEach(function (t) {
      line("M" + svg._X(t) + " " + svg._b.y1 + "v-4", "grid", null, svg);
      node("text", { x: svg._X(t), y: h - 12, "text-anchor": "middle", text: fmt(t, 3) }, svg);
    });
    SERIES.forEach(function (s, i) {
      var y = rowY(svg, i, SERIES.length);
      node("text", { x: 0, y: y + 3, class: "value", text: s.short }, svg);
      line("M" + 34 + " " + y + "H" + svg._X(hi), "grid", null, svg);
      var ale = stats(s.id, "aleatoric_mean");
      if (ale) {
        line("M" + svg._X(0) + " " + (y + 6) + "H" + svg._X(ale.mean), "series", INK[s.wl], svg,
          { "stroke-width": 1.3 });
        line("M" + svg._X(ale.mean) + " " + (y + 2) + "v8", "tickmark", INK[s.wl], svg);
        node("text", { x: svg._X(hi) + 8, y: y + 10, class: "value",
          text: "aleatoric " + fmt(ale.mean, 4) }, svg);
      }
      var reason = struckReason(s.id, "epistemic_mean");
      if (reason) {
        struck(svg, svg._X(0) + 6, y - 6, 30);
        node("text", { x: svg._X(0) + 42, y: y - 3, text: "epistemic not defined" }, svg);
        svg.lastChild.setAttribute("title", reason);
        return;
      }
      var epi = stats(s.id, "epistemic_mean");
      if (epi) {
        line("M" + svg._X(0) + " " + (y - 6) + "H" + svg._X(epi.mean), "series", INK[s.wl], svg,
          { "stroke-width": 1.3 });
        node("text", { x: svg._X(hi) + 8, y: y - 2, class: "value",
          text: "epistemic " + fmt(epi.mean, 5) }, svg);
      }
    });
    var cap = document.createElement("figcaption");
    cap.innerHTML = "<span>Upper rule: epistemic · lower rule: aleatoric, one shared scale</span>" +
      "<span>Mean over the test split</span>";
    fig.appendChild(cap);
    table(host, { rows: IDS, caption: "Uncertainty and error detection · three seeds",
      keys: ["epistemic_mean", "aleatoric_mean", "err_error_auroc", "ale_error_auroc"] });
  };

  /* ============================================================ VI evidence anatomy */
  var view = { model: "A1_fedl", image: 0 };

  FILL.anatomy = function (host) {
    note(host, "Forty-eight images from the frozen test split: fourteen picked by the reduction for " +
      "their position in the uncertainty field &mdash; confident and correct, hesitant and correct, " +
      "confident and wrong, wrong with high epistemic, maximal in either term, near-tied &mdash; and " +
      "the rest random. Selecting an image is one lookup in the exported posterior: no model runs in " +
      "the browser. These are the measurements of seed " + M.experiment.seeds[0] + ", whose checkpoints " +
      "were exported; the averaged tables in entries I and VII cover all three seeds.");
    controlBar(host, "Configuration", SERIES.map(function (s) {
      return { label: s.short + " " + s.name, value: s.id };
    }), function (v) { view.model = v; paint(host); }, view.model);

    var gal = document.createElement("div");
    gal.className = "gallery";
    gal.setAttribute("role", "group");
    gal.setAttribute("aria-label", "Test images in index order");
    S.images.forEach(function (im, i) {
      var b = document.createElement("button");
      b.type = "button";
      b.setAttribute("aria-pressed", i === view.image ? "true" : "false");
      b.title = im.id + " · true class " + im.label_name + " · " + im.selection + " pick";
      var img = document.createElement("img");
      img.src = "data/samples/" + im.id + ".png";
      img.alt = "CIFAR-10 test image " + im.index + ", true class " + im.label_name;
      img.width = 32; img.height = 32;
      var flag = document.createElement("span");
      flag.className = "flag";
      flag.textContent = "miss";
      b.appendChild(img);
      b.appendChild(flag);
      b.addEventListener("click", function () {
        view.image = i;
        Array.prototype.forEach.call(gal.children, function (o, j) {
          o.setAttribute("aria-pressed", j === i ? "true" : "false");
        });
        paint(host);
      });
      gal.appendChild(b);
    });
    host.appendChild(gal);
    var stage = document.createElement("div");
    host.appendChild(stage);
    host._gal = gal;
    host._stage = stage;
    paint(host);
    small(host, "Greyed thumbnails with a <em>miss</em> flag are images the selected configuration " +
      "classified wrongly. Amber marks the true class in the plates; the series colour marks the " +
      "prediction.");
  };

  function paint(host) {
    var stage = host._stage, gal = host._gal;
    if (!stage) return;
    var im = S.images[view.image];
    var p = S[view.model];
    var s = byId[view.model];
    var iv = S.A1_evidential_internals;
    var pred = p.predicted_class[view.image];
    stage.textContent = "";
    Array.prototype.forEach.call(gal.children, function (b, j) {
      b.classList.toggle("miss", p.predicted_class[j] !== S.images[j].label);
    });

    var grid = document.createElement("div");
    grid.className = "anatomy";
    stage.appendChild(grid);

    var spec = document.createElement("div");
    spec.className = "anatomy__specimen";
    var img = document.createElement("img");
    img.src = "data/samples/" + im.id + ".png";
    img.alt = "Selected CIFAR-10 test image, true class " + im.label_name;
    spec.appendChild(img);
    var rd = document.createElement("dl");
    rd.className = "readout";
    var rows = [["True class", im.label_name],
                ["Predicted", S.class_names[pred] + (pred === im.label ? " · correct" : " · miss")],
                ["Confidence, highest mean", fmt(p.mean[view.image][pred], 5)],
                ["Epistemic", struckReason(view.model, "epistemic_mean") ? "not defined"
                  : fmt(p.epistemic[view.image], 6)],
                ["Aleatoric", fmt(p.aleatoric[view.image], 5)],
                ["Monte-Carlo draws", String(p.n_monte_carlo)]];
    if (view.model === "A1_fedl") {
      rows.push(["Tau from the third head", fmt(iv.tau[view.image], 4)]);
      rows.push(["Alpha-nought, total evidence", fmt(iv.alpha0[view.image], 3)]);
    }
    rows.forEach(function (r) {
      var d = document.createElement("div");
      var dt = document.createElement("dt");
      dt.textContent = r[0];
      var dd = document.createElement("dd");
      dd.textContent = r[1];
      d.appendChild(dt);
      d.appendChild(dd);
      rd.appendChild(d);
    });
    spec.appendChild(rd);
    grid.appendChild(spec);

    var right = document.createElement("div");
    grid.appendChild(right);
    var two = document.createElement("div");
    two.className = "plates";
    right.appendChild(two);

    var w = 300, h = 256;
    var box = { x0: 20, x1: w - 12, y0: 40, y1: h - 62, lo: -0.5, hi: 9.5, ylo: 0, yhi: 1 };
    function classLabel(svg, x, k) {
      node("text", { transform: "rotate(-90 " + x + " " + (h - 58) + ")", x: x,
        y: h - 58, "text-anchor": "end", text: S.class_names[k] }, svg);
    }

    /* Plate A: the Dirichlet evidence vector, which only the evidential head produces. */
    var figA = document.createElement("figure");
    two.appendChild(figA);
    if (view.model !== "A1_fedl") {
      note(figA, "<strong>Plate A &mdash; Dirichlet evidence.</strong> Struck: this configuration has " +
        "no alpha head, so there is no evidence vector to draw. Its predictive mean in plate B comes from " +
        (view.model === "A3_laplace_icla" ? "the Laplace posterior over the last layer."
          : "the softmax of a single point estimate."));
    } else {
      var ev = iv.evidence[view.image];
      var maxEv = Math.max.apply(null, ev.concat([1e-6]));
      var svgA = plate(figA, w, h, box, "Per-class Dirichlet evidence for the selected image, seed " +
        M.experiment.seeds[0]);
      svgA.setAttribute("class", "state-partial");
      line("M" + box.x0 + " " + svgA._Y(0) + "H" + box.x1, "axis", null, svgA);
      node("text", { x: 0, y: 12, text: "alpha minus one, largest " + fmt(maxEv, 1) }, svgA);
      ev.forEach(function (e, k) {
        var x = svgA._X(k);
        var v = e / maxEv;
        line("M" + x + " " + svgA._Y(0) + "V" + svgA._Y(Math.max(v, 0.006)), "series",
          INK[k === im.label ? CURSOR : s.wl], svgA,
          { "stroke-opacity": k === im.label || k === pred ? 1 : 0.45, "stroke-width": 1.3 });
        classLabel(svgA, x, k);
        if (e > 0.05) {
          node("text", { x: x, y: svgA._Y(Math.max(v, 0.006)) - 4, "text-anchor": "middle",
            text: fmt(e, 1) }, svgA);
        }
      });
      var capA = document.createElement("figcaption");
      capA.innerHTML = "<span>Plate A · evidence, alpha minus one</span><span>amber marks the true class</span>";
      figA.appendChild(capA);
    }

    /* Plate B: predictive mean with its posterior standard deviation. */
    var figB = document.createElement("figure");
    two.appendChild(figB);
    var svgB = plate(figB, w, h, box, "Per-class predictive mean for the selected image, seed " + M.experiment.seeds[0]);
    svgB.setAttribute("class", "state-partial");
    line("M" + box.x0 + " " + svgB._Y(0) + "H" + box.x1, "axis", null, svgB);
    p.mean[view.image].forEach(function (mu, k) {
      var x = svgB._X(k);
      var sd = Math.sqrt(Math.max(p.variance[view.image][k], 0));
      var shown = k === im.label || k === pred;
      line("M" + x + " " + svgB._Y(0) + "V" + svgB._Y(Math.max(mu, 0.004)), "series",
        INK[k === im.label ? CURSOR : s.wl], svgB,
        { "stroke-opacity": shown ? 1 : 0.4, "stroke-width": 1.3 });
      var top = svgB._Y(Math.min(1, mu + sd));
      var bot = svgB._Y(Math.max(0, mu - sd));
      if (mu > 0.02 && sd > 1e-6) {
        line("M" + (x - 5) + " " + top + "h10M" + (x - 5) + " " + bot + "h10", "spread", null, svgB);
      }
      classLabel(svgB, x, k);
      if (shown) {
        node("text", { x: x, y: svgB._Y(Math.max(mu, 0.004)) - 5, "text-anchor": "middle",
          class: "value", text: fmt(mu, 3) }, svgB);
      }
    });
    var capB = document.createElement("figcaption");
    capB.innerHTML = "<span>Plate B · predictive mean</span><span>ticks = one standard deviation</span>";
    figB.appendChild(capB);

    small(right, "The mean and variance in plate B are the flexible-Dirichlet moments of the same " +
      "alpha, p and tau triple drawn in plate A; epistemic is the variance summed over classes and " +
      "aleatoric is one minus the sum of squared means, both computed in " +
      "<code>src/fd_distribution.py</code> and exported per image by " +
      "<code>src/export_webapp_data.py</code>. Both plates are linear, so one confident class takes " +
      "the whole rule and the rest sit on the axis: that is what a decisive posterior looks like, and " +
      "the struck epistemic row in entry V is what its absence looks like.");
  }

  /* ============================================================ VII seed spread */
  var seedState = { metric: "ece" };
  var SEED_KEYS = ["top1", "nll", "ece", "aurc", "err_error_auroc", "epistemic_mean",
                   "near_svhn_total_auroc"];

  FILL.seeds = function (host) {
    note(host, "Three independent training runs on the same frozen split. A mean without this view " +
      "invites a one-seed effect to be read as a method effect; the marks below are the runs, and the " +
      "pale rule is their mean.");
    var holder = document.createElement("div");
    controlBar(host, "Quantity", SEED_KEYS.map(function (k) {
      return { label: labelOf(k), value: k };
    }), function (v) { seedState.metric = v; holder.textContent = ""; drawSeeds(holder); },
    seedState.metric);
    host.appendChild(holder);
    drawSeeds(holder);
  };

  function drawSeeds(host) {
    var key = seedState.metric, seeds = M.experiment.seeds;
    var vals = [];
    SERIES.forEach(function (s) {
      seeds.forEach(function (sd) {
        var v = seedValue(sd, s.id, key);
        if (v !== null) vals.push(v);
      });
    });
    var lo = Math.min.apply(null, vals), hi = Math.max.apply(null, vals);
    var pad = ((hi - lo) || 0.01) * 0.22;
    lo = Math.max(0, lo - pad); hi = hi + pad;   /* no quantity here can be negative */
    var fig = document.createElement("figure");
    host.appendChild(fig);
    var w = 660, h = 190;
    var svg = plate(fig, w, h, { x0: 52, x1: w - 156, y0: 18, y1: h - 34, lo: lo, hi: hi,
      ylo: 0, yhi: SERIES.length }, labelOf(key) + " per seed, five configurations");
    [lo + (hi-lo)*0.08, (lo + hi) / 2, hi - (hi-lo)*0.08].forEach(function (t) {
      line("M" + svg._X(t) + " " + svg._b.y1 + "v-4", "grid", null, svg);
      node("text", { x: svg._X(t), y: h - 12, "text-anchor": "middle",
        text: fmt(t, dpOf(key)) }, svg);
    });
    SERIES.forEach(function (s, i) {
      var y = rowY(svg, i, SERIES.length);
      node("text", { x: 0, y: y + 3, class: "value", text: s.short }, svg);
      line("M" + 34 + " " + y + "H" + svg._X(hi), "grid", null, svg);
      if (struckReason(s.id, key)) {
        struck(svg, svg._b.x0 + 6, y, 30);
        node("text", { x: svg._b.x0 + 42, y: y + 3, text: "not defined" }, svg);
        svg.lastChild.setAttribute("title", s.why);
        return;
      }
      var st = M.models[s.id][key];
      line("M" + svg._X(st.mean) + " " + (y - 10) + "V" + (y + 10), "grid", INK[s.wl], svg,
        { "stroke-opacity": 0.5 });
      seeds.forEach(function (sd) {
        var v = seedValue(sd, s.id, key);
        if (v === null) return;
        line("M" + (svg._X(v) - 4) + " " + y + "h8", "tickmark", INK[s.wl], svg,
          { "stroke-width": 1.4 });
        node("title", { text: "seed " + sd + " · " + fmt(v, dpOf(key)) }, svg.lastChild);
      });
      node("text", { x: svg._X(hi) + 8, y: y + 3, class: "value",
        text: fmt(st.mean, dpOf(key)) + " ± " + fmt(st.std, dpOf(key)) }, svg);
    });
    var cap = document.createElement("figcaption");
    cap.innerHTML = "<span>" + labelOf(key) + " · each mark is one seed</span><span>n = " +
      seeds.length + "</span>";
    fig.appendChild(cap);

    var wrap = document.createElement("div");
    wrap.className = "table-wrap";
    var t = document.createElement("table");
    var head = "<tr><th scope='col' class='row-head'>Configuration</th>";
    seeds.forEach(function (sd) { head += "<th scope='col'>seed " + sd + "</th>"; });
    head += "<th scope='col'>mean</th><th scope='col'>std</th></tr>";
    var body = "";
    SERIES.forEach(function (s) {
      body += "<tr><th scope='row' class='row-head'>" + s.short + " · " + s.name + "</th>";
      if (struckReason(s.id, key)) {
        body += "<td colspan='" + (seeds.length + 2) + "' class='struck' style='text-align:left'>" +
          "not defined — " + s.why + "</td>";
      } else {
        seeds.forEach(function (sd) {
          var v = seedValue(sd, s.id, key);
          body += "<td class='num'>" + (v === null ? "—" : fmt(v, dpOf(key))) + "</td>";
        });
        body += "<td class='num'>" + fmt(M.models[s.id][key].mean, dpOf(key)) + "</td>" +
          "<td class='num'>" + fmt(M.models[s.id][key].std, dpOf(key)) + "</td>";
      }
      body += "</tr>";
    });
    t.innerHTML = "<caption>Every run, not their average</caption><thead>" + head + "</thead>" +
      "<tbody>" + body + "</tbody>";
    wrap.appendChild(t);
    host.appendChild(wrap);
  }

  /* ============================================================ VIII ablations */
  var varState = { variant: "identity", metric: "nll" };
  var VAR_KEYS = ["top1", "nll", "brier", "ece", "aurc", "err_error_auroc",
                  "near_svhn_total_auroc", "far_tiny_imagenet_total_auroc", "epistemic_mean"];
  var VAR_TEXT = {
    identity: "The main reduction. ICLA replaces the last-layer curvature with the identity matrix, so " +
      "the posterior is the prior alone plus a Gaussian on the logits, and only the prior precision lambda is fitted.",
    kron: "The same last-layer Laplace with a real Kronecker generalised-Gauss-Newton factor per class, " +
      "so curvature comes from the data and lambda trades off against it.",
    none: "Curvature dropped entirely: the posterior keeps only the prior, so predictive spread is set " +
      "by lambda and cannot respond to the training data.",
    identity_nosn: "Spectral normalisation removed from every convolutional and linear layer of " +
      "VGG-16-BN, taking away the per-layer Lipschitz budget both backbones were trained under. All " +
      "five configurations, including the baseline, share this setting, so the comparison against the " +
      "main run is like for like."
  };

  function duplicateOfMain(v) {
    if (v.is_main) return false;
    var same = true, seen = false;
    IDS.forEach(function (id) {
      VAR_KEYS.forEach(function (k) {
        var a = v.metrics[id] && v.metrics[id][k];
        var b = M.models[id] && M.models[id][k];
        if (!a || !b) return;
        seen = true;
        if (Math.abs(a.mean - b.mean) > 1e-12 || Math.abs(a.std - b.std) > 1e-12) same = false;
      });
    });
    return seen && same;
  }

  FILL.variants = function (host) {
    document.getElementById("mark-variants").textContent = M.variants.length +
      " recorded run variants · curvature and Lipschitz budget";
    legend(host);
    note(host, "Each variant is a complete rerun of the pipeline with exactly the named setting changed, " +
      "including its own refit of the post-hoc layer. The main run is solid, ablations are dashed, and a " +
      "variant that has not finished all seeds is flagged where it says how many it has.");
    var holder = document.createElement("div");
    controlBar(host, "Variant", M.variants.map(function (v) {
      return { label: v.label + " (" + v.seeds.length + " seeds" +
        (duplicateOfMain(v) ? ", identical to main" : "") + ")", value: v.key };
    }), function (v) { varState.variant = v; holder.textContent = ""; drawVariants(holder); },
    varState.variant);
    controlBar(host, "Quantity", VAR_KEYS.map(function (k) {
      return { label: labelOf(k), value: k };
    }), function (v) { varState.metric = v; holder.textContent = ""; drawVariants(holder); },
    varState.metric);
    host.appendChild(holder);
    drawVariants(holder);
  };

  function drawVariants(holder) {
    var v = null;
    M.variants.forEach(function (o) { if (o.key === varState.variant) v = o; });
    if (!v) return;
    var partial = v.seeds.length < M.experiment.seeds.length;
    note(holder, "<strong>" + v.label + "</strong> — " + (VAR_TEXT[v.key] || "") +
      " Source: <code>" + v.source + "</code>, " + v.seeds.length + " seeds, curvature " +
      v.curvature + ", spectral normalisation " + (v.spectral_norm ? "on" : "off") + ".");
    if (duplicateOfMain(v)) {
      note(holder, "<strong>This variant reproduces the main reduction exactly.</strong> In this " +
        "implementation the identity and no-curvature settings take the same branch in " +
        "<code>src/laplace_icla.py</code>, so the ablation tests nothing and the identical numbers " +
        "are its proof. The comparison that carries information is identity against kron below.");
    }
    if (partial) {
      note(holder, "<strong>Incomplete: " + v.seeds.length + " of " + M.experiment.seeds.length +
        " seeds.</strong> The value is drawn as a single run's measurement, and its spread is not " +
        "comparable with the main table's yet.");
    }
    table(holder, { rows: IDS, keys: VAR_KEYS, src: v.metrics, dashed: !v.is_main,
      caption: v.label + " · " + v.seeds.join(", ") });
    var key = varState.metric;
    var vals = IDS.map(function (id) {
      return v.metrics[id] && v.metrics[id][key] ? v.metrics[id][key] : null;
    });
    var real = vals.filter(function (x) { return x; }).map(function (x) { return x.mean; });
    var lo = Math.min.apply(null, real), hi = Math.max.apply(null, real);
    var fig = document.createElement("figure");
    holder.appendChild(fig);
    var w = 660, h = 176;
    var svg = plate(fig, w, h, { x0: 52, x1: w - 120, y0: 16, y1: h - 30,
      lo: lo - (hi - lo) * 0.3 - 1e-6, hi: hi + (hi - lo) * 0.3 + 1e-6, ylo: 0, yhi: IDS.length },
      labelOf(key) + " under " + v.label);
    SERIES.forEach(function (s, i) {
      var y = rowY(svg, i, IDS.length);
      node("text", { x: 0, y: y + 3, class: "value", text: s.short }, svg);
      line("M" + 34 + " " + y + "H" + svg._X(svg._b.hi), "grid", null, svg);
      if (struckReason(s.id, key)) {
        struck(svg, svg._b.x0 + 6, y, 30);
        node("text", { x: svg._b.x0 + 42, y: y + 3, text: "not defined" }, svg);
        return;
      }
      if (!vals[i]) return;
      var cls = "series" + (v.is_main ? "" : " state-ablation");
      line("M" + svg._X(vals[i].mean - vals[i].std) + " " + y + "H" +
           svg._X(vals[i].mean + vals[i].std), cls, INK[s.wl], svg);
      line("M" + svg._X(vals[i].mean) + " " + (y - 6) + "V" + (y + 6), "tickmark " + cls,
        INK[s.wl], svg);
      node("text", { x: svg._X(vals[i].mean) + 7, y: y - 8, class: "value",
        text: fmt(vals[i].mean, dpOf(key)) }, svg);
    });
    line("M" + svg._b.x0 + " " + svg._b.y1 + "H" + svg._b.x1, "axis", null, svg);
    for (var tk = 0; tk <= 4; tk++) {
      var tv = svg._b.lo + (svg._b.hi - svg._b.lo) * tk / 4;
      line("M" + svg._X(tv).toFixed(1) + " " + svg._b.y1 + "v4", "grid", null, svg);
      node("text", { x: svg._X(tv).toFixed(1), y: svg._b.y1 + 15, "text-anchor": "middle",
        text: fmt(tv, dpOf(key)) }, svg);
    }
    node("text", { x: svg._b.x0, y: h - 4, text: labelOf(key) }, svg);
    var cap = document.createElement("figcaption");
    cap.innerHTML = "<span>" + v.label + "</span><span>whisker = one standard deviation over " +
      v.seeds.length + " seeds</span>";
    fig.appendChild(cap);
    small(holder, "The comparison that matters is against the main run in entry I, not across " +
      "ablations: each variant refits its own lambda and its own calibrators, which is the point of the " +
      "ablation and also why the deltas are modest.");
  }

  /* ============================================================ IX instruments */
  FILL.instruments = function (host) {
    note(host, "None of these layers was trained together with the network. Each is fitted afterwards " +
      "on the validation portion of the training split, then applied to the frozen test split, so a " +
      "difference between B1 and B2, or B1 and A3, is a difference in the reduction and not in the " +
      "weights.");
    var seeds = M.experiment.seeds;
    var lams = seeds.map(function (sd) {
      var lap = perSeed(sd).laplace;
      return lap && typeof lap.lambda === "number" ? lap.lambda : null;
    });
    var closed = M.cost.map(function (c) { return c.lambda_closed_form; });
    var all = lams.concat(closed).filter(function (x) { return x; });
    /* The three fitted values sit within a few units of each other, so the scale is drawn around
       them rather than from zero, which would flatten every reading onto one mark. */
    var lo = Math.min.apply(null, all), hi = Math.max.apply(null, all);
    var span = (hi - lo) || hi * 0.05;
    lo -= span * 0.35; hi += span * 0.35;
    var fig = document.createElement("figure");
    host.appendChild(fig);
    var w = 660, h = 176;
    var svg = plate(fig, w, h, { x0: 84, x1: w - 190, y0: 18, y1: h - 44, lo: lo, hi: hi,
      ylo: 0, yhi: seeds.length }, "Fitted Laplace prior precision per seed");
    seeds.forEach(function (sd, i) {
      var y = rowY(svg, i, seeds.length);
      node("text", { x: 0, y: y + 3, class: "value", text: String(sd) }, svg);
      line("M" + svg._b.x0 + " " + y + "H" + svg._b.x1, "grid", null, svg);
      if (lams[i]) {
        line("M" + svg._X(lams[i]) + " " + (y - 9) + "V" + (y + 9), "tickmark", INK[436], svg,
          { "stroke-width": 1.4 });
        node("text", { x: svg._X(lams[i]) + 7, y: y - 11, class: "value",
          text: "λ " + fmt(lams[i], 2) }, svg);
      }
      if (closed[i]) {
        line("M" + svg._X(closed[i]) + " " + (y + 4) + "h16", "ref", INK[REF], svg);
        node("text", { x: svg._X(closed[i]) + 20, y: y + 8,
          text: "closed form " + fmt(closed[i], 2) }, svg);
      }
    });
    line("M" + svg._b.x0 + " " + svg._b.y1 + "H" + svg._b.x1, "axis", null, svg);
    for (var tk = 0; tk <= 3; tk++) {
      var tv = lo + (hi - lo) * tk / 3;
      line("M" + svg._X(tv).toFixed(1) + " " + svg._b.y1 + "v4", "grid", null, svg);
      node("text", { x: svg._X(tv).toFixed(1), y: svg._b.y1 + 16, "text-anchor": "middle",
        text: fmt(tv, 1) }, svg);
    }
    var cap = document.createElement("figcaption");
    cap.innerHTML = "<span>Prior precision lambda, fitted by marginal-likelihood search</span>" +
      "<span>dotted = the closed form, P over the squared weight norm</span>";
    fig.appendChild(cap);
    note(host, "With identity curvature the search lands on the closed-form optimum to three " +
      "significant figures. That coincidence is the check that the golden-section fit does what the " +
      "ICLA recipe says it does; it is not an independent result.");

    var wrap = document.createElement("div");
    wrap.className = "table-wrap";
    var t = document.createElement("table");
    var head = "<tr><th scope='col' class='row-head'>Instrument reading</th>";
    seeds.forEach(function (sd) { head += "<th scope='col'>seed " + sd + "</th>"; });
    head += "</tr>";
    var fields = [["Lambda, prior precision", "lambda", 2],
                  ["log-likelihood at the MAP", "log_lik_map", 1],
                  ["log evidence", "evidence", 1],
                  ["Squared norm of the last-layer weights", "weight_norm_sq", 3],
                  ["feature dimension", "n_features", 0],
                  ["parameters in the posterior", "n_params", 0],
                  ["curvature trace", "curvature_trace", 0]];
    var body = "";
    fields.forEach(function (f) {
      body += "<tr><th scope='row' class='row-head'>" + f[0] + "</th>";
      seeds.forEach(function (sd) {
        var lap = perSeed(sd).laplace;
        var v = lap ? lap[f[1]] : null;
        body += "<td class='num'>" + (typeof v === "number" ? fmt(v, f[2]) : "—") + "</td>";
      });
      body += "</tr>";
    });
    t.innerHTML = "<caption>A3 · last-layer Laplace, identity curvature</caption><thead>" + head +
      "</thead><tbody>" + body + "</tbody>";
    wrap.appendChild(t);
    host.appendChild(wrap);
    small(host, "The curvature trace is zero because the identity recipe discards the curvature term; " +
      "the quantity that was removed is reported as removed instead of hidden.");

    sub(host, "Temperature and Dirichlet rescaling");
    var wrap2 = document.createElement("div");
    wrap2.className = "table-wrap";
    var t2 = document.createElement("table");
    var h2 = "<tr><th scope='col' class='row-head'>Fitted parameter</th>";
    seeds.forEach(function (sd) { h2 += "<th scope='col'>seed " + sd + "</th>"; });
    h2 += "</tr>";
    var readers = [
      ["B2 temperature T", function (c) {
        return c.b2_temperature ? fmt(c.b2_temperature.temperature, 4) : "—"; }],
      ["A2 Dirichlet scale t, mean", function (c) {
        return c.a2_dirichlet ? fmt(c.a2_dirichlet.t_mean, 4) : "—"; }],
      ["A2 t across classes", function (c) {
        return c.a2_dirichlet ? fmt(c.a2_dirichlet.t_min, 3) + " to " +
          fmt(c.a2_dirichlet.t_max, 3) : "—"; }],
      ["A2 bias magnitude", function (c) {
        return c.a2_dirichlet ? fmt(c.a2_dirichlet.b_absmax, 4) : "—"; }]];
    var b2 = "";
    readers.forEach(function (r) {
      b2 += "<tr><th scope='row' class='row-head'>" + r[0] + "</th>";
      seeds.forEach(function (sd) {
        b2 += "<td class='num'>" + r[1](perSeed(sd).calibrators || {}) + "</td>";
      });
      b2 += "</tr>";
    });
    t2.innerHTML = "<caption>Post-hoc calibrators, fitted on validation only</caption><thead>" + h2 +
      "</thead><tbody>" + b2 + "</tbody>";
    wrap2.appendChild(t2);
    host.appendChild(wrap2);
    note(host, "A single temperature is one number shared by every class; the Dirichlet rescaling is a " +
      "vector, and the class-to-class range printed above is what makes it a strictly wider family. All " +
      "three seeds fit above or below one, so the direction of the correction is not a coincidence of " +
      "one run.");
  };

  /* ============================================================ X cost and provenance */
  FILL.cost = function (host) {
    note(host, "Measured on the machine named in the approved plan: an RTX 4060 Laptop GPU with 8 GB of " +
      "video memory, in the same environment the report was written in.");
    var seeds = M.experiment.seeds;
    var wrap = document.createElement("div");
    wrap.className = "table-wrap";
    var t = document.createElement("table");
    var head = "<tr><th scope='col' class='row-head'>Measured</th>";
    seeds.forEach(function (sd) { head += "<th scope='col'>seed " + sd + "</th>"; });
    head += "</tr>";
    var rows = [
      ["Both backbones trained, seconds", function (c) { return fmt(c.training_seconds, 1); }],
      ["Mean epoch, B1 baseline", function (c) {
        return c.mean_epoch_seconds ? fmt(c.mean_epoch_seconds.b1, 1) + " s" : "—"; }],
      ["Mean epoch, A1 evidential", function (c) {
        return c.mean_epoch_seconds ? fmt(c.mean_epoch_seconds.a1, 1) + " s" : "—"; }],
      ["Epochs before the stop, B1", function (c) {
        return c.epochs_run ? c.epochs_run.b1 + " of 100" : "—"; }],
      ["Epochs before the stop, A1", function (c) {
        return c.epochs_run ? c.epochs_run.a1 + " of 100" : "—"; }],
      ["Peak video memory, B1", function (c) {
        return c.peak_vram_mb ? fmt(c.peak_vram_mb.b1, 0) + " MB" : "—"; }],
      ["Peak video memory, A1", function (c) {
        return c.peak_vram_mb ? fmt(c.peak_vram_mb.a1, 0) + " MB" : "—"; }]];
    var body = "";
    rows.forEach(function (r) {
      body += "<tr><th scope='row' class='row-head'>" + r[0] + "</th>";
      seeds.forEach(function (sd) {
        var c = null;
        M.cost.forEach(function (o) { if (o.seed === sd) c = o; });
        body += "<td class='num'>" + (c ? r[1](c) : "—") + "</td>";
      });
      body += "</tr>";
    });
    t.innerHTML = "<caption>Training cost per seed</caption><thead>" + head + "</thead><tbody>" +
      body + "</tbody>";
    wrap.appendChild(t);
    host.appendChild(wrap);
    note(host, "The whole matrix — three seeds, five configurations, both backbones, evaluation and the " +
      "curvature ablations — took <strong>" + fmt(M.experiment.total_seconds, 0) + " s</strong> (" +
      fmt(M.experiment.total_seconds / 3600, 2) + " h) on <code>" + M.experiment.device + "</code>, " +
      "with " + M.n_monte_carlo + " Monte-Carlo draws per predictive call and a frozen split of " +
      M.dataset.test_images.toLocaleString("en-GB") + " test images.");

    sub(host, "Limits a reader should hold");
    var ul = document.createElement("ul");
    ul.className = "note";
    [
      "One station. CIFAR-10 only, with SVHN and Tiny-ImageNet as negative controls. The plan's " +
        "cross-dataset check on EuroSAT was not run, so nothing here generalises the finding to a " +
        "second benchmark.",
      "Three seeds. A standard deviation over three runs is a spread, not a significance test: a gap " +
        "smaller than its whisker is not a result.",
      "A2's variance is not carried through its own calibration, so its epistemic row is struck rather " +
        "than zero, and its aleatoric value is the rescaled one.",
      "Ablations share the seed set but refit their own post-hoc layer, which is the point of the " +
        "ablation and also the reason their deltas are modest.",
      "No hand-drawn input. This register is a static reading of exported posteriors. Classifying a new " +
        "image means re-running <code>src/evaluate.py</code> with the trained checkpoint, and the page " +
        "says so instead of pretending otherwise."
    ].forEach(function (txt) {
      var li = document.createElement("li");
      li.innerHTML = txt;
      ul.appendChild(li);
    });
    host.appendChild(ul);

    sub(host, "Honest absences");
    note(host, "Nothing on this page was invented for appearance. There are no comparison methods " +
      "beyond the five configurations, no dataset beyond the three named, and no accuracy, runtime or " +
      "citation here that is not also in the files listed in the colophon below. Where a quantity does " +
      "not exist for a configuration the register strikes it; where a run is unfinished it says so.");
  };

  /* ============================================================ colophon */
  function buildColophon() {
    var esc = function (s) {
      return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");
    };
    colophon.innerHTML =
      "<div><h2>What this page is</h2><p>The visualisation and the source-code link required by " +
      "components (3) and (6) of <em>CISC3024 AI Assignment 2</em>, Pattern Recognition, University of " +
      "Macau, Fall 2026. It is a reading of the experiment, not a second measurement: every figure is " +
      "loaded from <code>data/dataset.js</code>, generated by <code>src/export_webapp_data.py</code> " +
      "from the same JSON the report tables use.</p>" +
      "<p class='small'>Implementation by Qoder under the student's supervision; the student wrote no " +
      "code, as the brief requires, and the dialogue record is in " +
      "<code>research_records/conversation_log.md</code>.</p></div>" +
      "<div><h2>Provenance</h2><ul id='prov'></ul></div>" +
      "<div><h2>Reduction notes</h2><ul id='noteslist'></ul></div>" +
      "<div><h2>Reproduce</h2><ul>" +
      "<li><code>python src/prepare_data.py</code></li>" +
      "<li><code>python src/run_matrix.py --seeds 20261006 20261007 20261008</code></li>" +
      "<li><code>python src/export_webapp_data.py --results runs/full_*/matrix_results.json</code></li>" +
      "<li>open <code>webapp/index.html</code> directly &mdash; no server, no build step</li></ul>" +
      "<p class='small'>Typography: Archivo and JetBrains Mono, both variable, both SIL Open Font " +
      "Licence 1.1, subset and self-hosted in <code>webapp/fonts/</code>. Charts are hand-authored SVG " +
      "with no chart library.</p></div>";
    var prov = document.getElementById("prov");
    [M.source_results, "webapp/data/manifest.json", "webapp/data/samples.json",
     "data/splits.json", "runs/full_*/matrix_results*.json"].forEach(function (p) {
      var li = document.createElement("li");
      li.innerHTML = "<code>" + esc(p).replace(/\\+/g, "/") + "</code>";
      prov.appendChild(li);
    });
    var nl = document.getElementById("noteslist");
    (M.notes || []).forEach(function (txt) {
      var li = document.createElement("li");
      li.textContent = txt;
      nl.appendChild(li);
    });
  }

  /* ============================================================ rail and opening */
  var secs = Array.prototype.slice.call(entries.querySelectorAll(".entry"));

  function buildRail() {
    secs.forEach(function (sec) {
      var a = document.createElement("a");
      a.className = "tick";
      a.href = "#" + sec.id;
      node_text(a, "span", "tick__stub", "");
      node_text(a, "span", "tick__no", sec.id);
      node_text(a, "span", "tick__name", sec.querySelector(".entry__title").textContent);
      a.addEventListener("click", function (e) {
        e.preventDefault();
        open(sec.id, true);
        sec.scrollIntoView({ block: "start", behavior: scrollBehaviour() });
      });
      railBox.appendChild(a);
    });
  }
  function node_text(parent, tag, cls, text) {
    var el = document.createElement(tag);
    el.className = cls;
    el.textContent = text;
    parent.appendChild(el);
    return el;
  }

  function scrollBehaviour() {
    return window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches
      ? "auto" : "smooth";
  }

  function open(id, push) {
    secs.forEach(function (sec) {
      var on = sec.id === id;
      var was = sec.getAttribute("data-open") === "true";
      sec.setAttribute("data-open", on ? "true" : "false");
      if (on && !was) {
        sec.setAttribute("data-inking", "true");
        window.setTimeout(function () { sec.removeAttribute("data-inking"); }, 760);
      }
    });
    Array.prototype.forEach.call(railBox.children, function (t) {
      t.setAttribute("aria-current", t.getAttribute("href") === "#" + id ? "true" : "false");
    });
    if (push && history.replaceState) history.replaceState(null, "", "#" + id);
  }

  document.querySelectorAll(".entry__head").forEach(function (head) {
    head.addEventListener("click", function (e) {
      e.preventDefault();
      open(head.parentElement.id, true);
      head.parentElement.scrollIntoView({ block: "start", behavior: scrollBehaviour() });
    });
  });

  var firstSeed = M.experiment.seeds[0];
  [["mark-reliability", "One plate per configuration · seed "],
   ["mark-risk", "100 coverage steps · seed "],
   ["mark-anatomy", "Per-class quantities · seed "]].forEach(function (pair) {
    var el = document.getElementById(pair[0]);
    if (el) el.textContent = pair[1] + firstSeed + " only";
  });

  var params = document.getElementById("field-parameters");
  if (params && M.experiment.trainable_parameters) {
    params.textContent = M.experiment.trainable_parameters.toLocaleString("en-GB");
  }

  buildRail();
  Object.keys(FILL).forEach(function (k) {
    var host = entries.querySelector('[data-fill="' + k + '"]');
    if (host) FILL[k](host);
  });
  buildColophon();
  var initial = (window.location.hash || "#I").slice(1);
  if (!document.getElementById(initial)) initial = "I";
  open(initial, false);
  window.addEventListener("hashchange", function () {
    var id = (window.location.hash || "").slice(1);
    if (document.getElementById(id)) open(id, false);
  });
})();
