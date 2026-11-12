import type { InferenceSession } from "onnxruntime-web";
import type { FaceBox, MiniFasSample } from "@/types/vision";

const MODEL_URL = "/models/minifasnet/minifasnet-v2.onnx";
const INPUT_SIZE = 80;
const CROP_SCALE = 2.7;

type OrtModule = typeof import("onnxruntime-web/wasm");

let runtimePromise: Promise<OrtModule> | null = null;
let sessionPromise: Promise<InferenceSession> | null = null;
let inferenceQueue: Promise<void> = Promise.resolve();
let sourceCanvas: HTMLCanvasElement | null = null;

async function loadRuntime() {
  runtimePromise ??= import("onnxruntime-web/wasm").then((ort) => {
    ort.env.wasm.wasmPaths = "/wasm/ort/";
    ort.env.wasm.numThreads = 1;
    ort.env.wasm.proxy = false;
    return ort;
  });
  return runtimePromise;
}

export async function loadMiniFasModel() {
  sessionPromise ??= loadRuntime().then((ort) =>
    ort.InferenceSession.create(MODEL_URL, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all",
    }),
  );
  return sessionPromise;
}

function getScaledCrop(width: number, height: number, box: FaceBox) {
  const scale = Math.min(
    (height - 1) / box.height,
    (width - 1) / box.width,
    CROP_SCALE,
  );
  const cropWidth = box.width * scale;
  const cropHeight = box.height * scale;
  const centerX = box.x + box.width / 2;
  const centerY = box.y + box.height / 2;
  let left = centerX - cropWidth / 2;
  let top = centerY - cropHeight / 2;
  let right = centerX + cropWidth / 2;
  let bottom = centerY + cropHeight / 2;

  if (left < 0) {
    right -= left;
    left = 0;
  }
  if (top < 0) {
    bottom -= top;
    top = 0;
  }
  if (right > width - 1) {
    left -= right - width + 1;
    right = width - 1;
  }
  if (bottom > height - 1) {
    top -= bottom - height + 1;
    bottom = height - 1;
  }

  return {
    left: Math.max(0, Math.trunc(left)),
    top: Math.max(0, Math.trunc(top)),
    right: Math.min(width - 1, Math.trunc(right)),
    bottom: Math.min(height - 1, Math.trunc(bottom)),
  };
}

function readChannel(
  pixels: Uint8ClampedArray,
  width: number,
  x: number,
  y: number,
  channel: number,
) {
  return pixels[(y * width + x) * 4 + channel];
}

function preprocess(video: HTMLVideoElement, box: FaceBox) {
  const width = video.videoWidth;
  const height = video.videoHeight;
  if (!width || !height) throw new Error("La cámara aún no tiene imagen.");
  if (Math.min(box.width, box.height) < 64) {
    throw new Error("Acerca el rostro: MiniFASNetV2 requiere al menos 64 px.");
  }

  sourceCanvas ??= document.createElement("canvas");
  sourceCanvas.width = width;
  sourceCanvas.height = height;
  const context = sourceCanvas.getContext("2d", { willReadFrequently: true });
  if (!context) throw new Error("No se pudo preparar el cuadro de video.");
  context.drawImage(video, 0, 0, width, height);
  const { data } = context.getImageData(0, 0, width, height);
  const crop = getScaledCrop(width, height, box);
  const cropWidth = Math.max(1, crop.right - crop.left + 1);
  const cropHeight = Math.max(1, crop.bottom - crop.top + 1);
  const tensorData = new Float32Array(3 * INPUT_SIZE * INPUT_SIZE);
  const planeSize = INPUT_SIZE * INPUT_SIZE;

  for (let outputY = 0; outputY < INPUT_SIZE; outputY += 1) {
    const sourceY =
      crop.top + ((outputY + 0.5) * cropHeight) / INPUT_SIZE - 0.5;
    const yFloor = Math.floor(sourceY);
    const y0 = Math.max(crop.top, yFloor);
    const y1 = Math.min(crop.bottom, y0 + 1);
    const yWeight = sourceY - yFloor;
    for (let outputX = 0; outputX < INPUT_SIZE; outputX += 1) {
      const sourceX =
        crop.left + ((outputX + 0.5) * cropWidth) / INPUT_SIZE - 0.5;
      const xFloor = Math.floor(sourceX);
      const x0 = Math.max(crop.left, xFloor);
      const x1 = Math.min(crop.right, x0 + 1);
      const xWeight = sourceX - xFloor;
      const outputIndex = outputY * INPUT_SIZE + outputX;

      for (let outputChannel = 0; outputChannel < 3; outputChannel += 1) {
        const sourceChannel = 2 - outputChannel;
        const top =
          readChannel(data, width, x0, y0, sourceChannel) * (1 - xWeight) +
          readChannel(data, width, x1, y0, sourceChannel) * xWeight;
        const bottom =
          readChannel(data, width, x0, y1, sourceChannel) * (1 - xWeight) +
          readChannel(data, width, x1, y1, sourceChannel) * xWeight;
        tensorData[outputChannel * planeSize + outputIndex] =
          top * (1 - yWeight) + bottom * yWeight;
      }
    }
  }

  return tensorData;
}

function softmax(values: number[]) {
  const maximum = Math.max(...values);
  const exponentials = values.map((value) => Math.exp(value - maximum));
  const total = exponentials.reduce((sum, value) => sum + value, 0);
  return exponentials.map((value) => value / total);
}

async function runMiniFas(
  video: HTMLVideoElement,
  box: FaceBox,
): Promise<MiniFasSample> {
  const startedAt = performance.now();
  const [ort, session] = await Promise.all([loadRuntime(), loadMiniFasModel()]);
  const input = preprocess(video, box);
  const tensor = new ort.Tensor("float32", input, [1, 3, INPUT_SIZE, INPUT_SIZE]);
  const output = await session.run({ [session.inputNames[0]]: tensor });
  const logits = Array.from(output[session.outputNames[0]].data, Number);
  const probabilities = softmax(logits);
  const predictedClass = probabilities.indexOf(Math.max(...probabilities));

  return {
    realProbability: probabilities[1] ?? 0,
    spoofProbability: 1 - (probabilities[1] ?? 0),
    predictedClass,
    inferenceMs: performance.now() - startedAt,
  };
}

export function analyzeAntiSpoof(video: HTMLVideoElement, box: FaceBox) {
  const task = inferenceQueue.then(() => runMiniFas(video, box));
  inferenceQueue = task.then(
    () => undefined,
    () => undefined,
  );
  return task;
}
