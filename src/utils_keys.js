import * as ort from "onnxruntime-web/webgpu";
// utils.js

/**
 * Resize + pad image into square input size while keeping aspect ratio.
 * Returns {canvas, ratio, padX, padY}.
 */
export function letterbox(source, newSize = 224, color = [114, 114, 114]) {
  const iw = source.width;
  const ih = source.height;

  const scale = Math.min(newSize / iw, newSize / ih);
  const nw = Math.round(iw * scale);
  const nh = Math.round(ih * scale);

  const canvas = document.createElement("canvas");
  canvas.width = newSize;
  canvas.height = newSize;
  const ctx = canvas.getContext("2d");

  // Fill background
  ctx.fillStyle = `rgb(${color[0]},${color[1]},${color[2]})`;
  ctx.fillRect(0, 0, newSize, newSize);

  // Centered pad
  const dx = Math.floor((newSize - nw) / 2);
  const dy = Math.floor((newSize - nh) / 2);

  ctx.drawImage(source, 0, 0, iw, ih, dx, dy, nw, nh);

  return {
    canvas,
    ratio: scale,
    padX: dx,
    padY: dy,
  };
}

/**
 * Convert canvas image to Float32Array tensor in NCHW format.
 * Normalized to [0,1].
 */
export function toNchwFloat32(canvas) {
  const ctx = canvas.getContext("2d");
  const { width, height } = canvas;
  const imgData = ctx.getImageData(0, 0, width, height).data;

  // NHWC → NCHW
  const float32Data = new Float32Array(1 * 3 * height * width);
  let idx = 0;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const i = (y * width + x) * 4;
      const r = imgData[i] / 255.0;
      const g = imgData[i + 1] / 255.0;
      const b = imgData[i + 2] / 255.0;
      float32Data[idx] = r;                          // R
      float32Data[idx + height * width] = g;         // G
      float32Data[idx + 2 * height * width] = b;     // B
      idx++;
    }
  }

  return new ort.Tensor("float32", float32Data, [1, 3, height, width]);
}
