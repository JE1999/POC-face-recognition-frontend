import type { NormalizedLandmark } from "@mediapipe/tasks-vision";

export type LoadState = "idle" | "loading" | "ready" | "error";

export type FaceBox = {
  x: number;
  y: number;
  width: number;
  height: number;
};

export type FaceSample = {
  count: number;
  descriptor: Float32Array | null;
  box: FaceBox | null;
  inferenceMs: number;
};

export type IdentityDecision =
  "match" | "no-match" | "unstable" | "unavailable";
export type AntiSpoofDecision =
  | "real"
  | "spoof"
  | "unstable"
  | "unavailable"
  | "disabled";

export type MiniFasSample = {
  realProbability: number;
  spoofProbability: number;
  predictedClass: number;
  inferenceMs: number;
};

export type PhoneDecision =
  | "clear"
  | "detected"
  | "unstable"
  | "unavailable"
  | "disabled";

export type PhoneDetectionSample = {
  detected: boolean;
  confidence: number | null;
  box: FaceBox | null;
  inferenceMs: number;
};
export type LivenessAction = "blink" | "turn-left" | "turn-right" | "center";
export type LivenessStatus =
  | "not-started"
  | "waiting"
  | "verifying"
  | "action-complete"
  | "completed"
  | "failed"
  | "timed-out";

export type MotionMetrics = {
  faceCount: number;
  blinkScore: number;
  yaw: number;
  landmarks: NormalizedLandmark[];
  inferenceMs: number;
};

export type AdvancedSettings = {
  identityThreshold: number;
  antiSpoofEnabled: boolean;
  antiSpoofThreshold: number;
  phoneDetectionEnabled: boolean;
  phoneConfidenceThreshold: number;
  livenessEnabled: boolean;
  actionTimeoutSeconds: number;
  blinkThreshold: number;
  turnThreshold: number;
  showBoxes: boolean;
  showLandmarks: boolean;
  enabledActions: LivenessAction[];
};
