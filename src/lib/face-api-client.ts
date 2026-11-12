import type { FaceSample } from "@/types/vision";

type FaceApiModule = typeof import("@vladmandic/face-api");

let modulePromise: Promise<FaceApiModule> | null = null;
let modelsPromise: Promise<FaceApiModule> | null = null;

function getModule() {
  modulePromise ??= import("@vladmandic/face-api");
  return modulePromise;
}

export function loadFaceApiModels(): Promise<FaceApiModule> {
  modelsPromise ??= getModule().then(async (faceapi) => {
    await Promise.all([
      faceapi.nets.tinyFaceDetector.loadFromUri("/models/face-api"),
      faceapi.nets.faceLandmark68TinyNet.loadFromUri("/models/face-api"),
      faceapi.nets.faceRecognitionNet.loadFromUri("/models/face-api"),
    ]);
    return faceapi;
  });
  return modelsPromise;
}

export async function detectFaces(
  input: HTMLImageElement | HTMLVideoElement,
): Promise<FaceSample> {
  const faceapi = await loadFaceApiModels();
  const startedAt = performance.now();
  const detections = await faceapi
    .detectAllFaces(
      input,
      new faceapi.TinyFaceDetectorOptions({
        inputSize: 320,
        scoreThreshold: 0.5,
      }),
    )
    .withFaceLandmarks(true)
    .withFaceDescriptors();

  const first = detections[0];
  return {
    count: detections.length,
    descriptor: detections.length === 1 ? first.descriptor : null,
    box:
      detections.length === 1
        ? {
            x: first.detection.box.x,
            y: first.detection.box.y,
            width: first.detection.box.width,
            height: first.detection.box.height,
          }
        : null,
    inferenceMs: performance.now() - startedAt,
  };
}

export async function compareDescriptors(
  reference: Float32Array,
  candidate: Float32Array,
) {
  const faceapi = await getModule();
  return faceapi.euclideanDistance(reference, candidate);
}
