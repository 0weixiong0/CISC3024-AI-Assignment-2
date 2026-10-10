from pathlib import Path

# --- bench.js: the deep link can fire before the index arrives, so the class name was lost
p = Path("webapp/bench.js")
s = p.read_text(encoding="utf-8")

old = """  function loadSpecimenByIndex(i) {
    var probe = new Image();
    probe.onload = function () {
      imageToPlate(probe);
      specimenField.textContent = "test_" + String(i).padStart(5, "0") +
        " · true class " + (WALL.idx ? WALL.idx.records[i].c : "unknown");
      run();
    };"""
new = """  var plateIndex = null;

  function specimenLabel(i) {
    return "test_" + String(i).padStart(5, "0") + " · true class " +
      (WALL.idx ? WALL.idx.records[i].c : "reading the index…");
  }

  function loadSpecimenByIndex(i) {
    var probe = new Image();
    plateIndex = i;
    probe.onload = function () {
      imageToPlate(probe);
      specimenField.textContent = specimenLabel(i);
      run();
    };"""
assert old in s, "loadSpecimenByIndex"
s = s.replace(old, new, 1)

# once the index lands, refresh the field so a deep link does not stay on "reading the index"
old2 = """      var sentinel = document.getElementById("pick-more");"""
if old2 in s:
    s = s.replace(old2, """      if (plateIndex !== null) specimenField.textContent = specimenLabel(plateIndex);
      var sentinel = document.getElementById("pick-more");""", 1)
else:
    old3 = """      resort();
    }).catch(function (err) {
      document.getElementById("pick-count").textContent ="""
    assert old3 in s, "index ready anchor"
    s = s.replace(old3, """      if (plateIndex !== null) specimenField.textContent = specimenLabel(plateIndex);
      resort();
    }).catch(function (err) {
      document.getElementById("pick-count").textContent =""", 1)
p.write_text(s, encoding="utf-8")
print("bench.js patched")

# --- capture_webapp.py: the all-entries view needs the whole page, not one viewport
c = Path("src/capture_webapp.py")
t = c.read_text(encoding="utf-8")
t = t.replace("""        page.screenshot(path=str(OUT / name.replace("first_viewport", "all_entries")),
                        full_page=False)""",
              """        page.screenshot(path=str(OUT / name.replace("first_viewport", "all_entries")),
                        full_page=True)""")
c.write_text(t, encoding="utf-8")
print("capture script patched")
