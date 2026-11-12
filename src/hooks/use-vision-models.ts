"use client";

import { useCallback, useEffect, useState } from "react";
import { loadFaceApiModels } from "@/lib/face-api-client";
import {
  loadFaceLandmarker,
  loadObjectDetector,
} from "@/lib/mediapipe-client";
import { loadMiniFasModel } from "@/lib/minifasnet-client";
import type { LoadState } from "@/types/vision";

export function useVisionModels() {
  const [faceApiState, setFaceApiState] = useState<LoadState>("idle");
  const [mediaPipeState, setMediaPipeState] = useState<LoadState>("idle");
  const [miniFasState, setMiniFasState] = useState<LoadState>("idle");
  const [objectDetectorState, setObjectDetectorState] =
    useState<LoadState>("idle");
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async () => {
    setError(null);
    setFaceApiState("loading");
    setMediaPipeState("loading");
    setMiniFasState("loading");
    setObjectDetectorState("loading");
    const [faceResult, mediaResult, miniFasResult, objectResult] =
      await Promise.allSettled([
        loadFaceApiModels(),
        loadFaceLandmarker(),
        loadMiniFasModel(),
        loadObjectDetector(),
      ]);
    setFaceApiState(faceResult.status === "fulfilled" ? "ready" : "error");
    setMediaPipeState(mediaResult.status === "fulfilled" ? "ready" : "error");
    setMiniFasState(
      miniFasResult.status === "fulfilled" ? "ready" : "error",
    );
    setObjectDetectorState(
      objectResult.status === "fulfilled" ? "ready" : "error",
    );
    if (faceResult.status === "rejected")
      console.error("Face API model load failed", faceResult.reason);
    if (mediaResult.status === "rejected")
      console.error("MediaPipe model load failed", mediaResult.reason);
    if (miniFasResult.status === "rejected")
      console.error("MiniFASNet model load failed", miniFasResult.reason);
    if (objectResult.status === "rejected")
      console.error("MediaPipe Object Detector load failed", objectResult.reason);
    if (
      faceResult.status === "rejected" ||
      mediaResult.status === "rejected" ||
      miniFasResult.status === "rejected" ||
      objectResult.status === "rejected"
    ) {
      setError(
        "No se pudieron cargar todos los modelos locales. Revisa la consola de red y vuelve a intentar.",
      );
    }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => void load(), 0);
    return () => window.clearTimeout(timer);
  }, [load]);

  return {
    faceApiState,
    mediaPipeState,
    miniFasState,
    objectDetectorState,
    error,
    retry: load,
  };
}
