import * as ort from "onnxruntime-web/webgpu";
import { COCO_LABELS } from "./classes";

/**
 * Letterbox resize (preserve aspect ratio, pad to square).
 * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement} imageOrCanvas
 * @param {number} newSize square size (e.g., 640)
 * @param {number} padValue grayscale pad value (default 114)
 * @returns {{canvas: HTMLCanvasElement, scale: number, padLeft: number, padTop: number, nw: number, nh: number}}
 */
export function letterbox(imageOrCanvas, newSize, padValue = 114) {
  const iw = imageOrCanvas.videoWidth || imageOrCanvas.naturalWidth || imageOrCanvas.width;
  const ih = imageOrCanvas.videoHeight || imageOrCanvas.naturalHeight || imageOrCanvas.height;
  const scale = Math.min(newSize / iw, newSize / ih);
  const nw = Math.round(iw * scale);
  const nh = Math.round(ih * scale);
  const padW = newSize - nw;
  const padH = newSize - nh;
  const padLeft = Math.floor(padW / 2);
  const padTop = Math.floor(padH / 2);

  const c = document.createElement("canvas");
  c.width = newSize;
  c.height = newSize;
  const ctx = c.getContext("2d");

  ctx.fillStyle = `rgb(${padValue},${padValue},${padValue})`;
  ctx.fillRect(0, 0, newSize, newSize);
  ctx.drawImage(imageOrCanvas, padLeft, padTop, nw, nh);
  return { canvas: c, scale, padLeft, padTop, nw, nh };
}

/**
 * Convert a canvas to NCHW Float32 tensor in RGB, normalized to [0,1].
 * @param {HTMLCanvasElement} imgCanvas
 * @returns {ort.Tensor}
 */
export function toNchwFloat32(imgCanvas) {
  const width = imgCanvas.width;
  const height = imgCanvas.height;
  const ctx = imgCanvas.getContext("2d");
  const data = ctx.getImageData(0, 0, width, height).data;
  const size = width * height;
  const out = new Float32Array(3 * size);
  for (let i = 0; i < size; i++) {
    out[i] = data[i * 4] / 255;         // R
    out[i + size] = data[i * 4 + 1] / 255; // G
    out[i + 2 * size] = data[i * 4 + 2] / 255; // B
  }
  return new ort.Tensor("float32", out, [1, 3, height, width]);
}

/** IoU between two boxes {x1,y1,x2,y2}. */
export function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1);
  const y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2);
  const y2 = Math.min(a.y2, b.y2);
  const w = Math.max(0, x2 - x1);
  const h = Math.max(0, y2 - y1);
  const inter = w * h;
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / Math.max(1e-6, areaA + areaB - inter);
}

/**
 * Class-wise Non-Maximum Suppression.
 * @param {Array<{x1:number,y1:number,x2:number,y2:number,score:number,cls:number}>} dets
 * @param {number} iouThresh
 */
export function nonMaxSuppression(detections, iouThreshold = 0.45, maxDet = 100) {
  if (!detections || detections.length === 0) return [];

  // Sort by score descending
  detections.sort((a, b) => b.score - a.score);

  const keep = [];

  while (detections.length > 0 && keep.length < maxDet) {
    const best = detections.shift(); // highest score
    keep.push(best);

    detections = detections.filter(det => {
      if (det.clsId !== best.clsId) return true; // class-agnostic? remove this line
      return iou(best, det) < iouThreshold;
    });
  }

  return keep;
}

/**
 * Parse Ultralytics YOLO outputs (v5/v8/v11). Auto-detects layout.
 * Accepts shapes [1,N,S], [1,S,N], or [N,S].
 */
export function parseYolo(output, confThresh, inputSize) {
  const data = output.cpuData || output.data;
  const dims = output.dims || [];

  let numPred = 0;
  let stride = 0;
  let transposed = false;

  if (dims.length === 3) {
    if (dims[1] > dims[2]) { // [1, N, S]
      numPred = dims[1];
      stride = dims[2];
    } else {                 // [1, S, N]
      numPred = dims[2];
      stride = dims[1];
      transposed = true;
    }
  } else if (dims.length === 2) {
    numPred = dims[0];
    stride = dims[1];
  } else {
    throw new Error(`Unsupported output dims: ${dims.join("x")}`);
  }

  const likelyV8V11 = (stride === 84) || (stride !== 85);
  const getAt = (i, k) => (transposed ? data[k * numPred + i] : data[i * stride + k]);

  const dets = [];
  for (let i = 0; i < numPred; i++) {
    const cx = getAt(i, 0), cy = getAt(i, 1), w = getAt(i, 2), h = getAt(i, 3);

    let score = 0, bestCls = -1;
    if (likelyV8V11) {
      const C = stride - 4;
      for (let c = 0; c < C; c++) {
        const s = getAt(i, 4 + c);
        if (s > score) { score = s; bestCls = c; }
      }
    } else {
      const obj = getAt(i, 4);
      const C = stride - 5;
      let best = 0, bcls = -1;
      for (let c = 0; c < C; c++) {
        const s = getAt(i, 5 + c);
        if (s > best) { best = s; bcls = c; }
      }
      score = obj * best;
      bestCls = bcls;
    }

    if (score < confThresh) continue;

    const x1 = cx - w / 2;
    const y1 = cy - h / 2;
    const x2 = cx + w / 2;
    const y2 = cy + h / 2;

    dets.push({ x1, y1, x2, y2, score, cls: bestCls });
  }

  // If coords look normalized (<= 2), scale up to inputSize.
  const maxCoord = dets.reduce((m, d) => Math.max(m, d.x2, d.y2), 0);
  if (maxCoord <= 2.0) {
    for (const d of dets) {
      d.x1 *= inputSize; d.y1 *= inputSize; d.x2 *= inputSize; d.y2 *= inputSize;
    }
  }

  return dets;
}

/**
 * Draw detections on a target canvas using the source media as background.
 * @param {*} dets detections from parse + NMS
 * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement} source
 * @param {{padLeft:number,padTop:number,scale:number}} map letterbox mapping
 * @param {HTMLCanvasElement} canvas target canvas to draw on
 */
export function drawDetectionsOnSource(dets, source, map, canvas, opts = {}) {
  const { overlay = false } = opts;
  const ctx = canvas.getContext("2d");

  const sW = source.videoWidth || source.naturalWidth || source.width;
  const sH = source.videoHeight || source.naturalHeight || source.height;

  // Always keep the canvas pixel size in sync with the source frame size
  if (canvas.width !== sW || canvas.height !== sH) {
    canvas.width = sW;
    canvas.height = sH;
  }

  if (!overlay) {
    // Draw the frame when not overlaying (e.g. still image or hidden webcam video)
    ctx.drawImage(source, 0, 0, canvas.width, canvas.height);
  } else {
    // Transparent background for overlay; clear previous boxes
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  ctx.lineWidth = Math.max(2, Math.min(canvas.width, canvas.height) / 300);
  ctx.font = `${Math.max(12, Math.min(canvas.width, canvas.height) / 35)}px sans-serif`;

  dets.forEach((d) => {
    const x1 = Math.max(0, (d.x1 - map.padLeft) / map.scale);
    const y1 = Math.max(0, (d.y1 - map.padTop) / map.scale);
    const x2 = Math.min(canvas.width, (d.x2 - map.padLeft) / map.scale);
    const y2 = Math.min(canvas.height, (d.y2 - map.padTop) / map.scale);

    ctx.strokeStyle = "#00FF00";
    ctx.fillStyle = "rgba(0,0,0,0.5)";

    ctx.beginPath();
    ctx.rect(x1, y1, x2 - x1, y2 - y1);
    ctx.stroke();

    const label = `${COCO_LABELS[d.cls] || `cls ${d.cls}`} ${(d.score * 100).toFixed(1)}%`;
    const textW = ctx.measureText(label).width + 8;
    const textH = parseInt(ctx.font, 10) + 6;
    const tx = Math.max(0, Math.min(x1, canvas.width - textW));
    const ty = Math.max(textH, y1);
    ctx.fillRect(tx, ty - textH, textW, textH);
    ctx.fillStyle = "#FFFFFF";
    ctx.fillText(label, tx + 4, ty - 6);
  });
}


const MASK_THRESHOLD = 0.5; // tweak if needed
const PALETTE = [
  [255, 0, 0, 100],
  [0, 255, 0, 100],
  [0, 0, 255, 100],
  [255, 255, 0, 100],
  [255, 0, 255, 100],
  [0, 255, 255, 100],
];

/**
 * Draw YOLOv8-seg masks cropped to each detection bbox and mapped back to source size.
 *
 * @param {Array} dets  - kept detections [{x,y,w,h,clsId,score,maskCoeffs}, ...] in MODEL INPUT coords (letterboxed 0..inputSize)
 * @param {Array} masks - [{mask: Float32Array, width: pW, height: pH}, ...] raw proto-space masks (0..1) per det
 * @param {Object} lb   - letterbox info from your preprocess: expects lb.canvas, and (ratio, dw, dh) if available
 * @param {HTMLImageElement|HTMLVideoElement|HTMLCanvasElement} src - original source (image/video)
 * @param {HTMLCanvasElement} targetCanvas - where to paint
 */
export function drawYoloSegMasksCropped(dets, masks, lb, src, targetCanvas) {
  const ctx = targetCanvas.getContext("2d");
  const srcW = src.naturalWidth || src.videoWidth || src.width;
  const srcH = src.naturalHeight || src.videoHeight || src.height;

  // ensure output canvas matches source pixels
  if (targetCanvas.width !== srcW || targetCanvas.height !== srcH) {
    targetCanvas.width = srcW;
    targetCanvas.height = srcH;
  }

  // Draw the source frame first
  ctx.clearRect(0, 0, targetCanvas.width, targetCanvas.height);
  ctx.drawImage(src, 0, 0, srcW, srcH);

  // Model input (letterboxed) size
  const inputW = lb?.canvas?.width || 640;
  const inputH = lb?.canvas?.height || 640;

  // Undo-letterbox params (fallbacks if lb didn’t expose them)
  const ratio =
    lb?.ratio ??
    Math.min(inputW / srcW, inputH / srcH);
  const newUnpaddedW = Math.round(srcW * ratio);
  const newUnpaddedH = Math.round(srcH * ratio);
  const dw = lb?.dw ?? (inputW - newUnpaddedW) / 2;
  const dh = lb?.dh ?? (inputH - newUnpaddedH) / 2;

  for (let k = 0; k < dets.length; k++) {
    const det = dets[k];
    const { mask, width: pW, height: pH } = masks[k];

    // Build a proto-space RGBA image once per det (transparent outside mask)
    const rgba = new Uint8ClampedArray(pW * pH * 4);
    const [r, g, b, a] = PALETTE[det.clsId % PALETTE.length];

    for (let i = 0; i < pW * pH; i++) {
      const m = mask[i]; // 0..1
      const alpha = m > MASK_THRESHOLD ? a : 0;
      const j = i * 4;
      rgba[j + 0] = r;
      rgba[j + 1] = g;
      rgba[j + 2] = b;
      rgba[j + 3] = alpha;
    }

    const protoCanvas = document.createElement("canvas");
    protoCanvas.width = pW;
    protoCanvas.height = pH;
    const pctx = protoCanvas.getContext("2d");
    pctx.putImageData(new ImageData(rgba, pW, pH), 0, 0);

    // Box in model-input coords
    const x1 = det.x - det.w / 2;
    const y1 = det.y - det.h / 2;
    const w = det.w;
    const h = det.h;

    // Crop window in proto-space (160x160 typically)
    const sx = Math.max(0, Math.floor((x1 / inputW) * pW));
    const sy = Math.max(0, Math.floor((y1 / inputH) * pH));
    const sw = Math.max(1, Math.min(pW - sx, Math.ceil((w / inputW) * pW)));
    const sh = Math.max(1, Math.min(pH - sy, Math.ceil((h / inputH) * pH)));

    // Destination rect in ORIGINAL image pixels (undo letterbox)
    const dx = Math.round((x1 - dw) / ratio);
    const dy = Math.round((y1 - dh) / ratio);
    const dwPx = Math.round(w / ratio);
    const dhPx = Math.round(h / ratio);

    // Draw cropped & scaled mask into the box area only
    ctx.drawImage(protoCanvas, sx, sy, sw, sh, dx, dy, dwPx, dhPx);

    // Optional: outline box
    ctx.strokeStyle = `rgba(${r},${g},${b},0.9)`;
    ctx.lineWidth = 2;
    ctx.strokeRect(dx, dy, dwPx, dhPx);
  }
}