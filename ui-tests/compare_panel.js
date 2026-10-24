// Renders the dashboard in jsdom and drives the object-detection UI with real
// API payload shapes, asserting the side-by-side input/output panel and that the
// analytics view does not duplicate the media.
//
//   npm install --no-save jsdom
//   node ui-tests/compare_panel.js dashboard.html
//
// The dashboard is plain HTML + one inline script, so this exercises the real
// render path without a browser.
const fs = require("fs");
const assert = require("assert");
const { JSDOM } = require("jsdom");

const HTML = fs.readFileSync(process.argv[2] || "dashboard.html", "utf8");
const JPEG = "/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0a" +
  "HBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPDcxNDH/wAALCAABAAEBAREA/8QAFQABAQAAAAAA" +
  "AAAAAAAAAAAACAn/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFBEBAAAAAAAAAAAAAAAAAAAAAP/a" +
  "AAwDAQACEQMRAD8AmAA=";

const results = [];
function check(name, fn) {
  try {
    fn();
    results.push([name, true, ""]);
    console.log("PASS " + name);
  } catch (error) {
    results.push([name, false, error.message]);
    console.log("FAIL " + name + " :: " + error.message);
  }
}

const payloadFor = {
  image: {
    filename: "sample.png",
    media_type: "image",
    object_detection: {
      detections: [
        { class: "person", confidence: 0.8742, bbox: { x1: 10, y1: 20, x2: 110, y2: 220 } },
        { class: "person", confidence: 0.6613, bbox: { x1: 200, y1: 30, x2: 260, y2: 230 } },
        { class: "car", confidence: 0.5121, bbox: { x1: 300, y1: 150, x2: 480, y2: 260 } },
      ],
      detected_object_counts: { person: 2, car: 1 },
      // Real payloads carry >512 chars of base64, which is what marks the
      // field as an encoded image for the analytics renderer.
      annotated_image_base64: JPEG + "A".repeat(400),
      frames_analyzed: 1,
    },
  },
  video: {
    filename: "clip.mp4",
    media_type: "video",
    object_detection: {
      detections: [{ class: "car", confidence: 0.91, bbox: { x1: 1, y1: 2, x2: 30, y2: 40 }, frame_index: 12, timestamp_sec: 0.48 }],
      detected_object_counts: { car: 12 },
      tracks: [{ track_id: 1, class: "car", first_seen_sec: 0.1, last_seen_sec: 2.6, max_confidence: 0.91 }],
      annotated_video_url: "/object-videos/detected_abc_h264.mp4",
      annotated_video_download_url: "/object-videos/detected_abc_h264.mp4?download=1",
      annotated_video_codec: "h264",
      annotated_video_size_bytes: 217611,
      frames_analyzed: 75,
      frames_annotated: 75,
      duration_sec: 3,
      fps: 25,
    },
  },
};

function build() {
  const dom = new JSDOM(HTML, {
    runScripts: "dangerously",
    pretendToBeVisual: true,
    url: "http://127.0.0.1:5009/",
    beforeParse(window) {
      window.fetch = async (url) => ({
        ok: true,
        status: 200,
        json: async () => (String(url).includes("monitor")
          ? { available: true, device: { name: "GPU", total_mib: 1, used_mib: 1, used_percent: 1, free_percent: 99 }, processes: [], resource: { status: "ok", detail: "ok" } }
          : url.includes("config") ? { configuration: {} } : { jobs: [], page: { total: 0 } }),
        text: async () => "",
      });
      window.URL.createObjectURL = () => "blob:test-input";
      window.HTMLCanvasElement.prototype.getContext = () => ({
        clearRect() {}, beginPath() {}, moveTo() {}, lineTo() {}, stroke() {}, fillText() {},
        setLineDash() {}, fillRect() {}, measureText: () => ({ width: 10 }), save() {}, restore() {},
        scale() {}, set strokeStyle(v) {}, set fillStyle(v) {}, set lineWidth(v) {}, set font(v) {},
      });
    },
  });
  return dom;
}

function tab(dom, mode) {
  const button = dom.window.document.querySelector(`.tab[data-mode="${mode}"]`);
  assert(button, `tab ${mode} exists`);
  button.dispatchEvent(new dom.window.Event("click", { bubbles: true }));
}

function pick(dom, name, type) {
  const file = new dom.window.File([new dom.window.TextEncoder().encode("x")], name, { type });
  dom.window.choose(file);
  return file;
}

const dom = build();
const doc = dom.window.document;
const compare = () => doc.querySelector("#compare");

// ---------------- image object detection ----------------
tab(dom, "objects");
check("image: comparison hidden before a result", () => assert(compare().hidden));
pick(dom, "sample.png", "image/png");
dom.window.showPayload(payloadFor.image, payloadFor.image);

check("image: comparison becomes visible", () => assert.strictEqual(compare().hidden, false));
check("image: two panels (input + output)", () => assert.strictEqual(compare().querySelectorAll("figure").length, 2));
check("image: panel titles", () => {
  const titles = [...compare().querySelectorAll(".compare-head b")].map((n) => n.textContent);
  assert.deepStrictEqual(titles, ["Input", "Detected output · annotated image"]);
});
check("image: input shows the uploaded file", () => {
  const img = compare().querySelectorAll("figure")[0].querySelector("img");
  assert(img, "input image element");
  assert.strictEqual(img.getAttribute("src"), "blob:test-input");
  assert.strictEqual(img.getAttribute("alt"), "sample.png");
});
check("image: output shows the annotated image", () => {
  const img = compare().querySelectorAll("figure")[1].querySelector("img");
  assert(img, "output image element");
  assert(img.getAttribute("src").startsWith("data:image/jpeg;base64,"));
});
check("image: output caption counts detections and classes", () => {
  const note = compare().querySelectorAll(".compare-head span")[1].textContent;
  assert(note.includes("3 detections"), note);
  assert(note.includes("2 classes"), note);
});
check("image: analytics still lists detections in a table", () => {
  const analytics = doc.querySelector("#analytics");
  assert.strictEqual(analytics.hidden, false);
  const text = analytics.textContent;
  assert(text.includes("person") && text.includes("car"), "classes rendered");
  assert(analytics.querySelector("table"), "table rendered");
});
check("image: media is not duplicated inside analytics", () => {
  const analytics = doc.querySelector("#analytics");
  assert.strictEqual(analytics.querySelectorAll("img").length, 0, "no inline annotated image");
  assert(analytics.textContent.includes("Shown side by side above"));
});
check("image: raw JSON still present", () => {
  const raw = doc.querySelector("#result").textContent;
  assert(raw.includes("\"annotated_image_base64\""));
});

// ---------------- video object detection ----------------
tab(dom, "objectsVideo");
check("video: comparison cleared on tab switch", () => assert(compare().hidden));
pick(dom, "clip.mp4", "video/mp4");
dom.window.showPayload(payloadFor.video, payloadFor.video);

check("video: two panels (input + output)", () => assert.strictEqual(compare().querySelectorAll("figure").length, 2));
check("video: panel titles", () => {
  const titles = [...compare().querySelectorAll(".compare-head b")].map((n) => n.textContent);
  assert.deepStrictEqual(titles, ["Input", "Detected output · annotated clip"]);
});
check("video: input is a player bound to the upload", () => {
  const video = compare().querySelectorAll("figure")[0].querySelector("video");
  assert(video, "input video element");
  assert.strictEqual(video.getAttribute("src"), "blob:test-input");
  assert(video.hasAttribute("controls"));
});
check("video: output is a player bound to the annotated clip", () => {
  const video = compare().querySelectorAll("figure")[1].querySelector("video");
  assert(video, "output video element");
  assert.strictEqual(video.getAttribute("src"), "/object-videos/detected_abc_h264.mp4");
  assert(video.hasAttribute("controls"));
});
check("video: output caption shows codec, frames and size", () => {
  const note = compare().querySelectorAll(".compare-head span")[1].textContent;
  assert(note.includes("H264"), note);
  assert(note.includes("75 frames annotated"), note);
  assert(note.includes("MB") || note.includes("KB"), note);
  assert(note.includes("1 track"), note);
});
check("video: download link for the annotated clip", () => {
  const link = compare().querySelector(".compare-foot a");
  assert(link, "download link");
  assert.strictEqual(link.getAttribute("href"), "/object-videos/detected_abc_h264.mp4?download=1");
});
check("video: analytics shows tracks and detections tables", () => {
  const analytics = doc.querySelector("#analytics");
  const text = analytics.textContent;
  assert(text.includes("Object tracks") || text.includes("Tracks"), "tracks KPI or table");
  assert(analytics.querySelectorAll("table").length >= 2, "detections + tracks tables");
});
check("video: no duplicate players inside analytics", () => {
  assert.strictEqual(doc.querySelector("#analytics").querySelectorAll("video").length, 0);
});

// ---------------- non-detection payload leaves the panel alone ----------------
tab(dom, "image");
pick(dom, "photo.png", "image/png");
dom.window.showPayload({ filename: "photo.png", media_type: "image", detected_activity: ["walking"], confidence: 0.8 }, {});
check("analysis payload: comparison stays hidden", () => assert(compare().hidden));

const failed = results.filter(([, ok]) => !ok);
console.log(`\n${results.length - failed.length}/${results.length} UI checks passed`);
if (failed.length) {
  console.log("FAILED: " + failed.map(([name]) => name).join(", "));
  process.exit(1);
}
process.exit(0);
