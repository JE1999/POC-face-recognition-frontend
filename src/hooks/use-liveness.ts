"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type {
  AdvancedSettings,
  LivenessAction,
  LivenessStatus,
  MotionMetrics,
} from "@/types/vision";

function buildSequence(enabled: LivenessAction[]) {
  const motions = enabled.filter((action) => action !== "center");
  const randomized: LivenessAction[] = [...motions]
    .sort(() => Math.random() - 0.5)
    .slice(0, 2);
  if (
    randomized.some((action) => action.startsWith("turn")) &&
    enabled.includes("center")
  ) {
    randomized.push("center");
  }
  return randomized.length ? randomized : enabled.slice(0, 1);
}

export const actionCopy: Record<LivenessAction, string> = {
  blink: "Parpadea para continuar",
  "turn-left": "Gira la cabeza a tu izquierda",
  "turn-right": "Gira la cabeza a tu derecha",
  center: "Vuelve al centro",
};

export function useLiveness(settings: AdvancedSettings) {
  const [status, setStatus] = useState<LivenessStatus>("not-started");
  const [sequence, setSequence] = useState<LivenessAction[]>([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [remainingMs, setRemainingMs] = useState(0);
  const [failureReason, setFailureReason] = useState<string | null>(null);
  const actionStartedAtRef = useRef(0);
  const baselineYawRef = useRef(0);
  const blinkPhaseRef = useRef<"need-open" | "need-close" | "need-reopen">(
    "need-open",
  );
  const lastYawRef = useRef(0);
  const statusRef = useRef<LivenessStatus>(status);
  const indexRef = useRef(0);
  const sequenceRef = useRef<LivenessAction[]>([]);

  useEffect(() => {
    statusRef.current = status;
  }, [status]);

  const resetAction = useCallback(
    (yaw: number) => {
      actionStartedAtRef.current = performance.now();
      baselineYawRef.current = yaw;
      blinkPhaseRef.current = "need-open";
      setRemainingMs(settings.actionTimeoutSeconds * 1000);
    },
    [settings.actionTimeoutSeconds],
  );

  const cancel = useCallback(() => {
    setStatus("not-started");
    statusRef.current = "not-started";
    setSequence([]);
    sequenceRef.current = [];
    setCurrentIndex(0);
    indexRef.current = 0;
    setRemainingMs(0);
    setFailureReason(null);
  }, []);

  const start = useCallback(() => {
    const next = buildSequence(settings.enabledActions);
    if (!next.length) {
      setFailureReason("Selecciona al menos un desafío en el panel avanzado.");
      setStatus("failed");
      statusRef.current = "failed";
      return;
    }
    setSequence(next);
    sequenceRef.current = next;
    setCurrentIndex(0);
    indexRef.current = 0;
    setFailureReason(null);
    setStatus("waiting");
    statusRef.current = "waiting";
    resetAction(lastYawRef.current);
  }, [resetAction, settings.enabledActions]);

  const fail = useCallback((reason: string, timedOut = false) => {
    setFailureReason(reason);
    setStatus(timedOut ? "timed-out" : "failed");
    statusRef.current = timedOut ? "timed-out" : "failed";
  }, []);

  const completeAction = useCallback(
    (yaw: number) => {
      setStatus("action-complete");
      statusRef.current = "action-complete";
      window.setTimeout(() => {
        const nextIndex = indexRef.current + 1;
        if (nextIndex >= sequenceRef.current.length) {
          setStatus("completed");
          statusRef.current = "completed";
          setRemainingMs(0);
          return;
        }
        indexRef.current = nextIndex;
        setCurrentIndex(nextIndex);
        setStatus("verifying");
        statusRef.current = "verifying";
        resetAction(yaw);
      }, 450);
    },
    [resetAction],
  );

  const consumeMotion = useCallback(
    (metrics: MotionMetrics) => {
      lastYawRef.current = metrics.yaw;
      if (!["waiting", "verifying"].includes(statusRef.current)) return;
      if (metrics.faceCount !== 1) {
        fail(
          metrics.faceCount > 1
            ? "El intento se invalidó porque apareció más de un rostro."
            : "El intento se invalidó porque se perdió el rostro.",
        );
        return;
      }

      if (statusRef.current === "waiting") {
        setStatus("verifying");
        statusRef.current = "verifying";
        resetAction(metrics.yaw);
        return;
      }

      const action = sequenceRef.current[indexRef.current];
      if (!action) return;
      if (action === "blink") {
        const openLimit = settings.blinkThreshold * 0.45;
        if (
          blinkPhaseRef.current === "need-open" &&
          metrics.blinkScore <= openLimit
        ) {
          blinkPhaseRef.current = "need-close";
        } else if (
          blinkPhaseRef.current === "need-close" &&
          metrics.blinkScore >= settings.blinkThreshold
        ) {
          blinkPhaseRef.current = "need-reopen";
        } else if (
          blinkPhaseRef.current === "need-reopen" &&
          metrics.blinkScore <= openLimit
        ) {
          completeAction(metrics.yaw);
        }
      }

      const delta = metrics.yaw - baselineYawRef.current;
      if (action === "turn-left" && delta >= settings.turnThreshold) {
        completeAction(metrics.yaw);
      }
      if (action === "turn-right" && delta <= -settings.turnThreshold) {
        completeAction(metrics.yaw);
      }
      if (
        action === "center" &&
        Math.abs(metrics.yaw) <= settings.turnThreshold * 0.45
      ) {
        completeAction(metrics.yaw);
      }
    },
    [
      completeAction,
      fail,
      resetAction,
      settings.blinkThreshold,
      settings.turnThreshold,
    ],
  );

  useEffect(() => {
    if (!["waiting", "verifying"].includes(status)) return;
    const timer = window.setInterval(() => {
      const remaining =
        settings.actionTimeoutSeconds * 1000 -
        (performance.now() - actionStartedAtRef.current);
      setRemainingMs(Math.max(0, remaining));
      if (remaining <= 0)
        fail("Se agotó el tiempo para completar la acción.", true);
    }, 100);
    return () => window.clearInterval(timer);
  }, [fail, settings.actionTimeoutSeconds, status]);

  return {
    status,
    sequence,
    currentIndex,
    currentAction: sequence[currentIndex] ?? null,
    remainingMs,
    failureReason,
    isRunning: ["waiting", "verifying", "action-complete"].includes(status),
    start,
    cancel,
    consumeMotion,
  };
}
