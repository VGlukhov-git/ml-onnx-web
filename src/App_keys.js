// OnnxYoloSegDemo.jsx
import React, { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web/webgpu";
import { letterbox, toNchwFloat32 } from "./utils";
import "./App.css";

//
// === Config ===
//
const NUM_CLASSES = 80;   // adjust to your dataset
const MASK_DIM = 32;      // proto channels, usually 32
const CONF_THRESHOLD = 0.5;
const IOU_THRESHOLD = 0.45;
const MAX_DET = 50;
const MASK_THRESHOLD = 0.5;

const PALETTE = [
  [255, 0, 0, 100],
  [0, 255, 0, 100],
  [0, 0, 255, 100],
  [255, 255, 0, 100],
  [255, 0, 255, 100],
  [0, 255, 255, 100],
];

//
// === NMS ===
//
function nonMaxSuppression(detections, iouThreshold = 0.45, maxDet = 100) {
  if (!detections.length) return [];
  detections.sort((a, b) => b.score - a.score);
  const keep = [];
  while (detections.length && keep.length < maxDet) {
    const best = detections.shift();
    keep.push(best);
    detections = detections.filter((det) => {
      if (det.clsId !== best.clsId) return true;
      return iou(best, det) < iouThreshold;
    });
  }
  return keep;
}
function iou(a, b) {
  const boxA = [a.x - a.w / 2, a.y - a.h / 2, a.x + a.w / 2, a.y + a.h / 2];
  const boxB = [b.x - b.w / 2, b.y - b.h / 2, b.x + b.w / 2, b.y + b.h / 2];
  const interX1 = Math.max(boxA[0], boxB[0]);
  const interY1 = Math.max(boxA[1], boxB[1]);
  const interX2 = Math.min(boxA[2], boxB[2]);
  const interY2 = Math.min(boxA[3], boxB[3]);
  const interW = Math.max(0, interX2 - interX1);
  const interH = Math.max(0, interY2 - interY1);
  const interArea = interW * interH;
  const areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1]);
  const areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1]);
  return interArea / (areaA + areaB - interArea + 1e-9);
}

//
// === Mask drawing ===
//
function drawYoloSegMasksCropped(dets, masks, lb, src, targetCanvas) {
  const ctx = targetCanvas.getContext("2d");
  const srcW = src.naturalWidth || src.videoWidth || src.width;
  const srcH = src.naturalHeight || src.videoHeight || src.height;

  if (targetCanvas.width !== srcW || targetCanvas.height !== srcH) {
    targetCanvas.width = srcW;
    targetCanvas.height = srcH;
  }

  ctx.clearRect(0, 0, targetCanvas.width, targetCanvas.height);
  ctx.drawImage(src, 0, 0, srcW, srcH);

  const inputW = lb?.canvas?.width || 640;
  const inputH = lb?.canvas?.height || 640;
  const ratio =
    lb?.ratio ?? Math.min(inputW / srcW, inputH / srcH);
  const newUnpaddedW = Math.round(srcW * ratio);
  const newUnpaddedH = Math.round(srcH * ratio);
  const dw = lb?.dw ?? (inputW - newUnpaddedW) / 2;
  const dh = lb?.dh ?? (inputH - newUnpaddedH) / 2;

  for (let k = 0; k < dets.length; k++) {
    const det = dets[k];
    const { mask, width: pW, height: pH } = masks[k];
    const rgba = new Uint8ClampedArray(pW * pH * 4);
    const [r, g, b, a] = PALETTE[det.clsId % PALETTE.length];

    for (let i = 0; i < pW * pH; i++) {
      const m = mask[i];
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
    protoCanvas.getContext("2d").putImageData(new ImageData(rgba, pW, pH), 0, 0);

    const x1 = det.x - det.w / 2;
    const y1 = det.y - det.h / 2;
    const w = det.w;
    const h = det.h;

    const sx = Math.max(0, Math.floor((x1 / inputW) * pW));
    const sy = Math.max(0, Math.floor((y1 / inputH) * pH));
    const sw = Math.max(1, Math.min(pW - sx, Math.ceil((w / inputW) * pW)));
    const sh = Math.max(1, Math.min(pH - sy, Math.ceil((h / inputH) * pH)));

    const dx = Math.round((x1 - dw) / ratio);
    const dy = Math.round((y1 - dh) / ratio);
    const dwPx = Math.round(w / ratio);
    const dhPx = Math.round(h / ratio);

    ctx.drawImage(protoCanvas, sx, sy, sw, sh, dx, dy, dwPx, dhPx);

    ctx.strokeStyle = `rgba(${r},${g},${b},0.9)`;
    ctx.lineWidth = 2;
    ctx.strokeRect(dx, dy, dwPx, dhPx);
  }
}

//
// === Component ===
//
export default function OnnxYoloSegDemo() {
  const [session, setSession] = useState(null);
  const [usingWebGPU, setUsingWebGPU] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState("");
  const [modelFileName, setModelFileName] = useState("");
  const [imageFileName, setImageFileName] = useState("");
  const [inputSize, setInputSize] = useState(640);

  const imgRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  const camRunningRef = useRef(false);
  const rafRef = useRef(0);
  const lastInferRef = useRef(0);
  const inferIntervalMsRef = useRef(150); // throttle live inference
  const frameCanvasRef = useRef(null);
  const inferInFlightRef = useRef(false);

  useEffect(() => {
    ort.env.logLevel = "error";
    ort.env.wasm.wasmPaths =
      "https://cdn.jsdelivr.net/npm/onnxruntime-web@latest/dist/";
    if (ort.env && ort.env.webgpu) {
      ort.env.webgpu.powerPreference = "high-performance";
    }
    return () => stopCamera();
  }, []);

  async function loadOnnxModelFromFile(file) {
    setBusy(true);
    setMessage("Loading model…");
    try {
      const buffer = await file.arrayBuffer();
      const providers = navigator.gpu ? ["webgpu", "wasm"] : ["wasm"];
      const s = await ort.InferenceSession.create(buffer, { executionProviders: providers });
      setSession(s);
      setModelFileName(file.name);
      setUsingWebGPU(!!navigator.gpu);
      setMessage(`Model loaded. Outputs: ${s.outputNames.join(", ")}`);
    } catch (err) {
      console.error(err);
      setMessage(`Failed: ${err.message}`);
    } finally {
      setBusy(false);
    }
  }

  function onModelFileChange(e) {
    const f = e.target.files?.[0];
    if (f) loadOnnxModelFromFile(f);
  }

  function onImageFileChange(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    setImageFileName(f.name);
    const url = URL.createObjectURL(f);
    const img = imgRef.current;
    img.onload = () => {
      URL.revokeObjectURL(url);
      const canvas = canvasRef.current;
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;
      canvas.getContext("2d").drawImage(img, 0, 0);
    };
    img.src = url;
  }

  async function runSegmentationOnImage() {
    if (!session) return setMessage("Load an ONNX model first.");
    const img = imgRef.current;
    if (!img || !img.complete) return setMessage("Choose an image first.");
    setBusy(true);
    setMessage("Running…");
    try {
      const lb = letterbox(img, inputSize);
      const inputTensor = toNchwFloat32(lb.canvas);
      const results = await session.run({ [session.inputNames[0]]: inputTensor });
      const detOut = results[session.outputNames[0]];
      const protoOut = results[session.outputNames[1]];

      const dets = [];
      const [_, nAttr, nAnchors] = detOut.dims;
      for (let a = 0; a < nAnchors; a++) {
        const base = a * nAttr;
        const x = detOut.data[base + 0];
        const y = detOut.data[base + 1];
        const w = detOut.data[base + 2];
        const h = detOut.data[base + 3];
        let maxScore = -Infinity, clsId = -1;
        for (let c = 0; c < NUM_CLASSES; c++) {
          const score = detOut.data[base + 4 + c];
          if (score > maxScore) { maxScore = score; clsId = c; }
        }
        if (maxScore < CONF_THRESHOLD) continue;
        const maskCoeffs = detOut.data.slice(base + 4 + NUM_CLASSES, base + nAttr);
        dets.push({ x, y, w, h, score: maxScore, clsId, maskCoeffs });
      }
      const kept = nonMaxSuppression(dets, IOU_THRESHOLD, MAX_DET);

      const [__, pC, pH, pW] = protoOut.dims;
      const proto = protoOut.data;
      const masks = kept.map(det => {
        const m = new Float32Array(pH * pW).fill(0);
        for (let c = 0; c < MASK_DIM; c++) {
          const coeff = det.maskCoeffs[c];
          for (let i = 0; i < pH * pW; i++) {
            m[i] += coeff * proto[c * pH * pW + i];
          }
        }
        for (let i = 0; i < m.length; i++) m[i] = 1 / (1 + Math.exp(-m[i]));
        return { mask: m, width: pW, height: pH };
      });

      drawYoloSegMasksCropped(kept, masks, lb, img, canvasRef.current);
      setMessage(`Detections: ${kept.length}`);
    } catch (err) {
      console.error(err);
      setMessage(`Error: ${err.message}`);
    } finally {
      setBusy(false);
    }
  }

  async function startCamera() {
    if (!session) return setMessage("Load a model first.");
    if (camRunningRef.current) return;
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      const video = videoRef.current;
      video.srcObject = stream;
      await video.play();
      camRunningRef.current = true;
      loopCamera();
      setMessage("Camera running…");
    } catch (err) {
      setMessage(`Camera error: ${err.message}`);
    }
  }

  function stopCamera() {
    camRunningRef.current = false;
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    const video = videoRef.current;
    if (video?.srcObject) {
      video.srcObject.getTracks().forEach(t => t.stop());
      video.srcObject = null;
    }
  }

  async function inferOneVideoFrame(video, targetCanvas) {
    if (inferInFlightRef.current) return;
    inferInFlightRef.current = true;
    try {
      const frameCanvas = frameCanvasRef.current || (frameCanvasRef.current = document.createElement("canvas"));
      frameCanvas.width = video.videoWidth;
      frameCanvas.height = video.videoHeight;
      frameCanvas.getContext("2d").drawImage(video, 0, 0);

      const lb = letterbox(frameCanvas, inputSize);
      const inputTensor = toNchwFloat32(lb.canvas);
      const results = await session.run({ [session.inputNames[0]]: inputTensor });
      const detOut = results[session.outputNames[0]];
      const protoOut = results[session.outputNames[1]];

      const dets = [];
      const [_, nAttr, nAnchors] = detOut.dims;
      for (let a = 0; a < nAnchors; a++) {
        const base = a * nAttr;
        const x = detOut.data[base + 0];
        const y = detOut.data[base + 1];
        const w = detOut.data[base + 2];
        const h = detOut.data[base + 3];
        let maxScore = -Infinity, clsId = -1;
        for (let c = 0; c < NUM_CLASSES; c++) {
          const score = detOut.data[base + 4 + c];
          if (score > maxScore) { maxScore = score; clsId = c; }
        }
        if (maxScore < CONF_THRESHOLD) continue;
        const maskCoeffs = detOut.data.slice(base + 4 + NUM_CLASSES, base + nAttr);
        dets.push({ x, y, w, h, score: maxScore, clsId, maskCoeffs });
      }
      const kept = nonMaxSuppression(dets, IOU_THRESHOLD, MAX_DET);

      const [__, pC, pH, pW] = protoOut.dims;
      const proto = protoOut.data;
      const masks = kept.map(det => {
        const m = new Float32Array(pH * pW).fill(0);
        for (let c = 0; c < MASK_DIM; c++) {
          const coeff = det.maskCoeffs[c];
          for (let i = 0; i < pH * pW; i++) {
            m[i] += coeff * proto[c * pH * pW + i];
          }
        }
        for (let i = 0; i < m.length; i++) m[i] = 1 / (1 + Math.exp(-m[i]));
        return { mask: m, width: pW, height: pH };
      });

      drawYoloSegMasksCropped(kept, masks, lb, video, targetCanvas);
    } catch (err) {
      console.error(err);
    } finally {
      inferInFlightRef.current = false;
    }
  }

  function loopCamera() {
    if (!camRunningRef.current) return;
    const video = videoRef.current;
    if (video.readyState >= 2) {
      const now = performance.now();
      if (now - lastInferRef.current >= inferIntervalMsRef.current && !inferInFlightRef.current) {
        lastInferRef.current = now;
        inferOneVideoFrame(video, canvasRef.current);
      }
    }
    rafRef.current = requestAnimationFrame(loopCamera);
  }

  //
  // === UI ===
  //
  return (
    <div className="container">
      <h1>ONNX YOLOv8-seg Demo</h1>
      <p>{usingWebGPU ? "WebGPU" : "WASM"} backend</p>
      <section className="grid grid-3">
        <div className="panel">
          <h2>1) Load model</h2>
          <input type="file" accept=".onnx" onChange={onModelFileChange} />
          {modelFileName && <p>Loaded: {modelFileName}</p>}
        </div>
        <div className="panel">
          <h2>2) Image</h2>
          <input type="file" accept="image/*" onChange={onImageFileChange} />
          {imageFileName && <p>Selected: {imageFileName}</p>}
          <button onClick={runSegmentationOnImage} disabled={busy}>
            {busy ? "Working…" : "Segment image"}
          </button>
        </div>
        <div className="panel">
          <h2>Settings</h2>
          <label>Input size <input type="number" value={inputSize} onChange={e=>setInputSize(+e.target.value)} /></label>
        </div>
      </section>
      <section className="grid grid-2">
        <div className="panel">
          <h2>3) Webcam</h2>
          <button onClick={startCamera}>Start</button>
          <button onClick={stopCamera}>Stop</button>
          <video ref={videoRef} playsInline muted className="hidden" />
        </div>
        <div className="panel">
          <h2>Output</h2>
          <p>{message}</p>
          <canvas ref={canvasRef} className="canvas" />
          <img ref={imgRef} alt="" className="hidden" />
        </div>
      </section>
    </div>
  );
}
