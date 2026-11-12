"use client";

import { useCallback, useEffect, useRef, useState } from "react";

export function useCamera() {
  const [isActive, setIsActive] = useState(false);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceId] = useState("");
  const [error, setError] = useState<string | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const stop = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setIsActive(false);
  }, []);

  const refreshDevices = useCallback(async () => {
    if (!navigator.mediaDevices?.enumerateDevices) return;
    const available = (await navigator.mediaDevices.enumerateDevices()).filter(
      (device) => device.kind === "videoinput",
    );
    setDevices(available);
    if (!deviceId && available[0]?.deviceId) setDeviceId(available[0].deviceId);
  }, [deviceId]);

  const start = useCallback(
    async (video: HTMLVideoElement | null, requestedDeviceId = deviceId) => {
      setError(null);
      if (!video || !navigator.mediaDevices?.getUserMedia) {
        setError("Este navegador no ofrece acceso compatible a la cámara.");
        return;
      }
      stop();
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          audio: false,
          video: {
            deviceId: requestedDeviceId ? { exact: requestedDeviceId } : undefined,
            facingMode: requestedDeviceId ? undefined : "user",
            width: { ideal: 960 },
            height: { ideal: 720 },
          },
        });
        streamRef.current = stream;
        video.srcObject = stream;
        await video.play();
        setIsActive(true);
        await refreshDevices();
      } catch (cause) {
        const name = cause instanceof DOMException ? cause.name : "";
        setError(
          name === "NotAllowedError"
            ? "Permiso de cámara denegado. Habilítalo en el navegador e intenta de nuevo."
            : name === "NotFoundError"
              ? "No se encontró una cámara disponible."
              : "No fue posible iniciar la cámara. Puede estar en uso por otra aplicación.",
        );
      }
    },
    [deviceId, refreshDevices, stop],
  );

  useEffect(() => stop, [stop]);

  return {
    isActive,
    devices,
    deviceId,
    error,
    setDeviceId,
    start,
    stop,
  };
}
