import type { FaceLandmarker, ObjectDetector } from "@mediapipe/tasks-vision";
import type { MotionMetrics, PhoneDetectionSample } from "@/types/vision";

let landmarkerPromise: Promise<FaceLandmarker> | null = null;
let objectDetectorPromise: Promise<ObjectDetector> | null = null;

export async function loadFaceLandmarker(): Promise<FaceLandmarker> {
  landmarkerPromise ??= import("@mediapipe/tasks-vision").then(
    async ({ FaceLandmarker, FilesetResolver }) => {
      const fileset = await FilesetResolver.forVisionTasks("/wasm");
      const options = {
        baseOptions: {
          modelAssetPath: "/models/mediapipe/face_landmarker.task",
          delegate: "GPU" as const,
        },
        runningMode: "VIDEO" as const,
        numFaces: 2,
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true,
        minFaceDetectionConfidence: 0.5,
        minFacePresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      };

      try {
        return await FaceLandmarker.createFromOptions(fileset, options);
      } catch {
        return FaceLandmarker.createFromOptions(fileset, {
          ...options,
          baseOptions: {
            modelAssetPath: "/models/mediapipe/face_landmarker.task",
            delegate: "CPU",
          },
        });
      }
    },
  );
  return landmarkerPromise;
}

export async function loadObjectDetector(): Promise<ObjectDetector> {
  objectDetectorPromise ??= import("@mediapipe/tasks-vision").then(
    async ({ FilesetResolver, ObjectDetector }) => {
      const fileset = await FilesetResolver.forVisionTasks("/wasm");
      const options = {
        baseOptions: {
          modelAssetPath: "/models/mediapipe/efficientdet_lite0.tflite",
          delegate: "GPU" as const,
        },
        runningMode: "VIDEO" as const,
        scoreThreshold: 0.1,
        maxResults: 5,
        categoryAllowlist: ["cell phone"],
      };

      try {
        return await ObjectDetector.createFromOptions(fileset, options);
      } catch {
        return ObjectDetector.createFromOptions(fileset, {
          ...options,
          baseOptions: {
            modelAssetPath: "/models/mediapipe/efficientdet_lite0.tflite",
            delegate: "CPU",
          },
        });
      }
    },
  );
  return objectDetectorPromise;
}

export async function analyzeMotion(
  video: HTMLVideoElement,
  timestamp: number,
): Promise<MotionMetrics> {
  const landmarker = await loadFaceLandmarker();
  const startedAt = performance.now();
  const result = landmarker.detectForVideo(video, timestamp);
  const landmarks = result.faceLandmarks[0] ?? [];
  const categories = result.faceBlendshapes[0]?.categories ?? [];
  const score = (name: string) =>
    categories.find((category) => category.categoryName === name)?.score ?? 0;

  let yaw = 0;
  if (landmarks.length > 454) {
    const leftCheek = landmarks[234].x;
    const rightCheek = landmarks[454].x;
    const center = (leftCheek + rightCheek) / 2;
    const halfWidth = Math.max(Math.abs(rightCheek - leftCheek) / 2, 0.001);
    yaw = (landmarks[1].x - center) / halfWidth;
  }

  return {
    faceCount: result.faceLandmarks.length,
    blinkScore: (score("eyeBlinkLeft") + score("eyeBlinkRight")) / 2,
    yaw,
    landmarks,
    inferenceMs: performance.now() - startedAt,
  };
}

export async function detectPhone(
  video: HTMLVideoElement,
  timestamp: number,
): Promise<PhoneDetectionSample> {
  const detector = await loadObjectDetector();
  const startedAt = performance.now();
  const result = detector.detectForVideo(video, timestamp);
  const candidates = result.detections
    .map((detection) => ({
      detection,
      category: detection.categories.find(
        (category) => category.categoryName === "cell phone",
      ),
    }))
    .filter((candidate) => candidate.category)
    .sort(
      (left, right) =>
        (right.category?.score ?? 0) - (left.category?.score ?? 0),
    );
  const best = candidates[0];
  const boundingBox = best?.detection.boundingBox;

  return {
    detected: Boolean(best),
    confidence: best?.category?.score ?? null,
    box: boundingBox
      ? {
          x: boundingBox.originX,
          y: boundingBox.originY,
          width: boundingBox.width,
          height: boundingBox.height,
        }
      : null,
    inferenceMs: performance.now() - startedAt,
  };
}
