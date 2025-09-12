// OnnxObjectDetectionDemo.jsx
import React, { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web/webgpu";
import {
  letterbox,
  toNchwFloat32,
  nonMaxSuppression,
  parseYolo,
  drawDetectionsOnSource,
} from "./utils";
import "./App.css";

export default function OnnxObjectDetectionDemo() {
  const [session, setSession] = useState(null);
  const [usingWebGPU, setUsingWebGPU] = useState(false);
  const [busy, setBusy] = useState(false);
  const [modelFileName, setModelFileName] = useState("");
  const [imageFileName, setImageFileName] = useState("");
  const [message, setMessage] = useState("");

  const [inputSize, setInputSize] = useState(640);
  const [confThreshold, setConfThreshold] = useState(0.25);
  const [iouThreshold, setIouThreshold] = useState(0.45);

  // Still image & webcam
  const imgRef = useRef(null);
  const videoRef = useRef(null);  // hidden webcam video
  const canvasRef = useRef(null); // main canvas for image/webcam

  // Video FILE mode (visible player with overlay canvas)
  const fileVideoRef = useRef(null);
  const fileCanvasRef = useRef(null);
  const [videoFileName, setVideoFileName] = useState("");
  const [fileStatus, setFileStatus] = useState("");
  const fileObjectUrlRef = useRef(null);

  // Webcam & file loops
  const camRunningRef = useRef(false);
  const fileRunningRef = useRef(false);
  const rafRef = useRef(0);
  const fileRafRef = useRef(0);

  // Throttle & offscreen capture
  const lastInferRef = useRef(0);
  const inferIntervalMsRef = useRef(15);
  const frameCanvasRef = useRef(null);

  // Single mutex for session.run()
  const inferInFlightRef = useRef(false);

  useEffect(() => {
    ort.env.wasm.wasmPaths =
      "https://cdn.jsdelivr.net/npm/onnxruntime-web@latest/dist/";
    if (ort.env && ort.env.webgpu) {
      ort.env.webgpu.powerPreference = "high-performance";
    }
    return () => {
      stopCamera();
      stopFileDetection();
      if (fileObjectUrlRef.current) {
        URL.revokeObjectURL(fileObjectUrlRef.current);
        fileObjectUrlRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // ===== Model loading =====
  async function loadOnnxModelFromFile(file) {
    setBusy(true);
    setMessage("Loading model…");
    try {
      const buffer = await file.arrayBuffer();
      const providers = ["webgpu", "wasm"];
      const s = await ort.InferenceSession.create(buffer, {
        executionProviders: providers,
      });
      setSession(s);
      setModelFileName(file.name);
      setUsingWebGPU(!!navigator.gpu);
      setMessage(
        `Model loaded. Inputs: ${s.inputNames.join(", ")} | Outputs: ${s.outputNames.join(", ")}`
      );
    } catch (err) {
      console.error(err);
      setMessage(`Failed to load model: ${err?.message || err}`);
      setSession(null);
    } finally {
      setBusy(false);
    }
  }

  // ===== Still image detection =====
  async function runDetection() {
    if (!session) return setMessage("Load an ONNX model first.");
    if (camRunningRef.current || fileRunningRef.current)
      return setMessage("Stop live detection before running on a still image.");

    const img = imgRef.current;
    if (!img || !img.complete) return setMessage("Choose an image first.");

    setBusy(true);
    setMessage("Running inference…");
    inferInFlightRef.current = true;
    try {
      const lb = letterbox(img, inputSize);
      const inputTensor = toNchwFloat32(lb.canvas);
      const results = await session.run({ [session.inputNames[0]]: inputTensor });
      const out = results[session.outputNames[0]];
      const kept = nonMaxSuppression(parseYolo(out, confThreshold, inputSize), iouThreshold);
      drawDetectionsOnSource(kept, img, lb, canvasRef.current, { overlay: false });
      setMessage(`Detections: ${kept.length}${usingWebGPU ? " (WebGPU)" : " (WASM)"}`);
    } catch (err) {
      console.error(err);
      setMessage(`Inference failed: ${err?.message || err}`);
    } finally {
      inferInFlightRef.current = false;
      setBusy(false);
    }
  }

  function onModelFileChange(e) {
    const f = e.target.files && e.target.files[0];
    if (f) loadOnnxModelFromFile(f);
  }

  function onImageFileChange(e) {
    const f = e.target.files && e.target.files[0];
    if (!f) return;
    setImageFileName(f.name);
    const url = URL.createObjectURL(f);
    const img = imgRef.current;
    img.onload = () => {
      URL.revokeObjectURL(url);
      const canvas = canvasRef.current;
      const w = img.naturalWidth || img.width;
      const h = img.naturalHeight || img.height;
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0);
    };
    img.src = url;
  }

  // ===== Video FILE handling (overlay boxes on top of the video) =====
  function onVideoFileChange(e) {
    const f = e.target.files && e.target.files[0];
    if (!f) return;
    if (camRunningRef.current) {
      setFileStatus("Stop the webcam before loading a video file.");
      return;
    }
    stopFileDetection();

    setVideoFileName(f.name);
    setFileStatus("Loading video…");

    if (fileObjectUrlRef.current) {
      URL.revokeObjectURL(fileObjectUrlRef.current);
      fileObjectUrlRef.current = null;
    }

    const url = URL.createObjectURL(f);
    fileObjectUrlRef.current = url;

    const v = fileVideoRef.current;
    v.src = url;
    v.load();

    v.onloadedmetadata = () => {
      setFileStatus(`Loaded: ${f.name} · ${Math.round(v.duration)}s · ${v.videoWidth}×${v.videoHeight}`);
      // Keep the overlay canvas pixel size exactly equal to the video’s intrinsic size
      const c = fileCanvasRef.current;
      if (v.videoWidth && v.videoHeight) {
        c.width = v.videoWidth;
        c.height = v.videoHeight;
      }
    };
    v.onerror = () => setFileStatus("Failed to load this video.");
    v.onended = () => {
      setFileStatus("Video ended.");
      stopFileDetection();
    };
  }

  async function startFileDetection() {
    if (!session) return setFileStatus("Load an ONNX model first.");
    if (fileRunningRef.current) return setFileStatus("Detection already running.");
    if (camRunningRef.current) return setFileStatus("Stop the webcam first.");

    const v = fileVideoRef.current;
    if (!v.src) return setFileStatus("Choose a video file first.");
    try {
      await v.play();
      if (fileRafRef.current) cancelAnimationFrame(fileRafRef.current);
      fileRunningRef.current = true;
      loopFileVideo();
      setFileStatus("Running detection on video…");
    } catch (err) {
      console.error(err);
      setFileStatus(`Cannot play video: ${err?.message || err}`);
    }
  }

  function stopFileDetection() {
    fileRunningRef.current = false;
    if (fileRafRef.current) cancelAnimationFrame(fileRafRef.current);
    try { fileVideoRef.current?.pause(); } catch {}
  }

  function loopFileVideo() {
    if (!fileRunningRef.current) return;
    const v = fileVideoRef.current;
    if (!v || v.readyState < 2) {
      fileRafRef.current = requestAnimationFrame(loopFileVideo);
      return;
    }
    const now = performance.now();
    if (now - lastInferRef.current >= inferIntervalMsRef.current && !inferInFlightRef.current) {
      lastInferRef.current = now;
      // overlay=true so we draw ONLY boxes on the transparent canvas over the video
      inferOneVideoFrame(v, fileCanvasRef.current, { overlay: true }).catch(console.error);
    }
    fileRafRef.current = requestAnimationFrame(loopFileVideo);
  }

  // ===== Webcam handling =====
  async function startCamera() {
    if (!session) return setMessage("Load an ONNX model first.");
    if (camRunningRef.current) return setMessage("Camera already running.");
    if (fileRunningRef.current) return setMessage("Stop video-file detection first.");
    if (!navigator.mediaDevices?.getUserMedia)
      return setMessage("getUserMedia not supported in this browser.");

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment" },
        audio: false,
      });
      const video = videoRef.current;
      video.srcObject = stream;
      await video.play();
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      camRunningRef.current = true;
      lastInferRef.current = 0;
      loopCamera();
      setMessage("Camera started. Running live detection…");
    } catch (err) {
      console.error(err);
      setMessage(`Camera error: ${err?.message || err}`);
    }
  }

  function stopCamera() {
    camRunningRef.current = false;
    if (rafRef.current) cancelAnimationFrame(rafRef.current);
    const video = videoRef.current;
    if (video?.srcObject) {
      video.srcObject.getTracks().forEach((t) => t.stop());
      video.srcObject = null;
    }
    setMessage("Camera stopped.");
  }

  // ===== Shared per-frame inference =====
  async function inferOneVideoFrame(video, targetCanvas, { overlay = false } = {}) {
    if (inferInFlightRef.current) return;
    inferInFlightRef.current = true;
    try {
      // Keep target canvas pixel size equal to the source every frame (defensive)
      const sW = video.videoWidth, sH = video.videoHeight;
      if (!sW || !sH) { inferInFlightRef.current = false; return; }
      if (targetCanvas.width !== sW || targetCanvas.height !== sH) {
        targetCanvas.width = sW; targetCanvas.height = sH;
      }

      let frameCanvas = frameCanvasRef.current || (frameCanvasRef.current = document.createElement("canvas"));
      if (frameCanvas.width !== sW || frameCanvas.height !== sH) {
        frameCanvas.width = sW; frameCanvas.height = sH;
      }
      const fctx = frameCanvas.getContext("2d", { willReadFrequently: true });
      fctx.drawImage(video, 0, 0, sW, sH);

      const lb = letterbox(frameCanvas, inputSize);
      const inputTensor = toNchwFloat32(lb.canvas);
      const results = await session.run({ [session.inputNames[0]]: inputTensor });
      const out = results[session.outputNames[0]];
      const dets = nonMaxSuppression(parseYolo(out, confThreshold, inputSize), iouThreshold);

      drawDetectionsOnSource(dets, video, lb, targetCanvas, { overlay });
    } catch (err) {
      console.error(err);
      if (overlay) setFileStatus(`Video inference failed: ${err?.message || err}`);
      else setMessage(`Video inference failed: ${err?.message || err}`);
    } finally {
      inferInFlightRef.current = false;
    }
  }

  function loopCamera() {
    if (!camRunningRef.current) return;
    const video = videoRef.current;
    if (!video || video.readyState < 2) {
      rafRef.current = requestAnimationFrame(loopCamera);
      return;
    }
    const now = performance.now();
    if (now - lastInferRef.current >= inferIntervalMsRef.current && !inferInFlightRef.current) {
      lastInferRef.current = now;
      // webcam path draws frame+boxes into its canvas (not overlay)
      inferOneVideoFrame(video, canvasRef.current, { overlay: false }).catch(console.error);
    }
    rafRef.current = requestAnimationFrame(loopCamera);
  }

  // ===== UI =====
  return (
    <div className="cyber-page">
      <div className="cyber-bg" />
      <div className="container">
        <header className="header">
          <h1 className="glitch" data-text="ONNX Object Detection">
            ONNX Object Detection
          </h1>
          <p className="sub">
            Load a model, detect on an image, your webcam, or a <strong>video file</strong>.
            Inference stays on-device using <span className="chip">{usingWebGPU ? "WebGPU" : "WASM"}</span>.
          </p>
        </header>

        {/* Model / Image / Settings */}
        <section className="grid grid-3">
          <div className="panel">
            <h2 className="panel-title">1) Load ONNX model</h2>
            <input type="file" accept=".onnx" onChange={onModelFileChange} className="input" />
            {modelFileName && <p className="badge">Loaded: {modelFileName}</p>}
            <p className="hint">Tip: YOLOv5/v8/v11 export at {inputSize}×{inputSize}.</p>
          </div>

          <div className="panel">
            <h2 className="panel-title">2) Run on image</h2>
            <input type="file" accept="image/*" onChange={onImageFileChange} className="input" />
            {imageFileName && <p className="badge">Selected: {imageFileName}</p>}
            <button disabled={busy} onClick={runDetection} className={`btn btn-primary ${busy ? "is-busy" : ""}`}>
              {busy ? "Working…" : "Detect on image"}
            </button>
          </div>

          <div className="panel">
            <h2 className="panel-title">Settings</h2>
            <label className="label">
              <span>Input size</span>
              <input
                type="number"
                value={inputSize}
                onChange={(e)=>setInputSize(parseInt(e.target.value || "640", 10))}
                className="input"
              />
            </label>
            <label className="label">
              <span>Confidence threshold</span>
              <input
                type="number"
                step="0.01"
                min="0"
                max="1"
                value={confThreshold}
                onChange={(e)=>setConfThreshold(parseFloat(e.target.value || "0.25"))}
                className="input"
              />
            </label>
            <label className="label">
              <span>IoU threshold (NMS)</span>
              <input
                type="number"
                step="0.01"
                min="0"
                max="1"
                value={iouThreshold}
                onChange={(e)=>setIouThreshold(parseFloat(e.target.value || "0.45"))}
                className="input"
              />
            </label>
            <p className="hint">
              Backend:{" "}
              <span className={`chip ${usingWebGPU ? "chip-cyan" : "chip-pink"}`}>
                {usingWebGPU ? "WebGPU" : "WASM"}
              </span>
            </p>
          </div>
        </section>

        {/* Webcam + Output */}
        <section className="grid grid-2">
          <div className="panel">
            <h2 className="panel-title">3) Live webcam</h2>
            <div className="row">
              <button onClick={startCamera} className="btn btn-success">Start camera</button>
              <button onClick={stopCamera} className="btn btn-secondary">Stop camera</button>
            </div>
            <p className="hint">Grant camera permission. On mobile, back camera is preferred.</p>
            <video ref={videoRef} playsInline muted className="hidden" />
          </div>

          <div className="panel panel-output">
            <div className="panel-head">
              <h2 className="panel-title">Output</h2>
              <span className="status">{message}</span>
            </div>
            <div className="canvas-wrap">
              <canvas ref={canvasRef} className="canvas" />
              <img ref={imgRef} alt="uploaded" className="hidden" />
            </div>
          </div>
        </section>

        {/* Video FILE mode with overlayed boxes */}
        <section className="panel panel-video">
          <h2 className="panel-title">4) Video file (overlay)</h2>
          <div className="row">
            <input className="input" type="file" accept="video/*" onChange={onVideoFileChange} />
            <button onClick={startFileDetection} className="btn btn-primary">Start detection</button>
            <button onClick={stopFileDetection} className="btn">Stop</button>
          </div>
          {videoFileName && <p className="badge">Selected: {videoFileName}</p>}

          <div className="video-overlay" /* container must be position:relative in CSS */>
            <video
              ref={fileVideoRef}
              className="video-tag"
              controls
              muted
              playsInline
              style={{ position: "relative", zIndex: 1 }}
            />
            {/* z-index ensures canvas paints above GPU-composited <video> in some browsers */}
            <canvas
              ref={fileCanvasRef}
              className="overlay-canvas"
              style={{ zIndex: 2, pointerEvents: "none" }}
            />
          </div>

          <div className="status mt8">{fileStatus}</div>
        </section>

        <footer className="foot">
          <p className="foot-note">
            Parser auto-detects Ultralytics YOLO <strong>v8/v11</strong> and classic <strong>v5</strong>.
            For smoother live runs, lower input size or raise the throttle to 150–200&nbsp;ms.
          </p>
        </footer>
      </div>
    </div>
  );
}
