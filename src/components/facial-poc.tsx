"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  AlertCircle,
  Camera,
  CameraOff,
  Check,
  CircleUserRound,
  Eye,
  FileImage,
  Gauge,
  Info,
  LoaderCircle,
  RefreshCw,
  ScanFace,
  Settings2,
  ShieldCheck,
  ShieldX,
  Smartphone,
  Upload,
  X,
} from "lucide-react";
import { actionCopy, useLiveness } from "@/hooks/use-liveness";
import { useCamera } from "@/hooks/use-camera";
import { useVisionModels } from "@/hooks/use-vision-models";
import { compareDescriptors, detectFaces } from "@/lib/face-api-client";
import { analyzeMotion, detectPhone } from "@/lib/mediapipe-client";
import { analyzeAntiSpoof } from "@/lib/minifasnet-client";
import { cn } from "@/lib/utils";
import type {
  AdvancedSettings,
  AntiSpoofDecision,
  FaceSample,
  IdentityDecision,
  LivenessAction,
  MiniFasSample,
  MotionMetrics,
  PhoneDecision,
  PhoneDetectionSample,
} from "@/types/vision";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { Progress } from "@/components/ui/progress";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Separator } from "@/components/ui/separator";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";

type ReferenceState =
  "waiting" | "processing" | "valid" | "no-face" | "multiple" | "error";

const PHONE_IMMEDIATE_THRESHOLD = 0.45;

const initialSettings: AdvancedSettings = {
  identityThreshold: 0.50,
  antiSpoofEnabled: true,
  antiSpoofThreshold: 0.5,
  phoneDetectionEnabled: true,
  phoneConfidenceThreshold: 0.22,
  livenessEnabled: true,
  actionTimeoutSeconds: 8,
  blinkThreshold: 0.52,
  turnThreshold: 0.22,
  showBoxes: true,
  showLandmarks: false,
  enabledActions: ["blink", "turn-left", "turn-right", "center"],
};

const emptyFaceSample: FaceSample = {
  count: 0,
  descriptor: null,
  box: null,
  inferenceMs: 0,
};

const referenceCopy: Record<ReferenceState, string> = {
  waiting: "Esperando fotografía",
  processing: "Analizando fotografía…",
  valid: "Referencia válida",
  "no-face": "No se detectó un rostro",
  multiple: "Se detectó más de un rostro",
  error: "No fue posible procesar la imagen",
};

function ModelBadge({
  label,
  state,
}: {
  label: string;
  state: "idle" | "loading" | "ready" | "error";
}) {
  return (
    <Badge
      variant="outline"
      className={cn(
        "gap-1.5 rounded-full border-white/10 bg-white/5 px-3 py-1 text-white/80",
        state === "ready" &&
        "border-emerald-400/20 bg-emerald-400/10 text-emerald-200",
        state === "error" && "border-rose-400/20 bg-rose-400/10 text-rose-200",
      )}
    >
      <span
        className={cn(
          "size-1.5 rounded-full bg-amber-300",
          state === "ready" && "bg-emerald-300",
          state === "error" && "bg-rose-300",
        )}
      />
      {label}:{" "}
      {state === "ready" ? "listo" : state === "error" ? "error" : "cargando"}
    </Badge>
  );
}

function ResultRow({
  label,
  value,
  tone,
  detail,
}: {
  label: string;
  value: string;
  tone: "success" | "danger" | "pending" | "neutral";
  detail?: string;
}) {
  return (
    <div className="flex items-center justify-between gap-4 py-3.5">
      <div>
        <p className="text-sm font-medium text-slate-800">{label}</p>
        {detail ? (
          <p className="mt-0.5 text-xs text-slate-500">{detail}</p>
        ) : null}
      </div>
      <Badge
        className={cn(
          "shrink-0 rounded-full px-3 py-1",
          tone === "success" && "bg-emerald-100 text-emerald-800",
          tone === "danger" && "bg-rose-100 text-rose-800",
          tone === "pending" && "bg-amber-100 text-amber-800",
          tone === "neutral" && "bg-slate-100 text-slate-700",
        )}
      >
        {value}
      </Badge>
    </div>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-xl border border-slate-200 bg-slate-50 px-3 py-2.5">
      <p className="text-[11px] font-medium uppercase tracking-[0.12em] text-slate-500">
        {label}
      </p>
      <p className="mt-1 font-mono text-sm font-semibold text-slate-900">
        {value}
      </p>
    </div>
  );
}

export function FacialPoc() {
  const {
    faceApiState,
    mediaPipeState,
    miniFasState,
    objectDetectorState,
    error: modelError,
    retry,
  } = useVisionModels();
  const camera = useCamera();
  const [settings, setSettings] = useState(initialSettings);
  const liveness = useLiveness(settings);
  const imageRef = useRef<HTMLImageElement>(null);
  const referenceCanvasRef = useRef<HTMLCanvasElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const cameraCanvasRef = useRef<HTMLCanvasElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const previousObjectUrlRef = useRef<string | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [fileName, setFileName] = useState("");
  const [referenceState, setReferenceState] =
    useState<ReferenceState>("waiting");
  const [referenceSample, setReferenceSample] =
    useState<FaceSample>(emptyFaceSample);
  const [cameraSample, setCameraSample] = useState<FaceSample>(emptyFaceSample);
  const [motionMetrics, setMotionMetrics] = useState<MotionMetrics | null>(
    null,
  );
  const [distance, setDistance] = useState<number | null>(null);
  const [identityDecision, setIdentityDecision] =
    useState<IdentityDecision>("unavailable");
  const [antiSpoofDecision, setAntiSpoofDecision] =
    useState<AntiSpoofDecision>("unavailable");
  const [miniFasSample, setMiniFasSample] =
    useState<MiniFasSample | null>(null);
  const [phoneDecision, setPhoneDecision] =
    useState<PhoneDecision>("unavailable");
  const [phoneSample, setPhoneSample] =
    useState<PhoneDetectionSample | null>(null);
  const comparisonHistoryRef = useRef<boolean[]>([]);
  const antiSpoofHistoryRef = useRef<number[]>([]);
  const phoneHistoryRef = useRef<boolean[]>([]);
  const phoneLatchedRef = useRef(false);
  const livenessConsumerRef = useRef(liveness.consumeMotion);
  const cancelLivenessRef = useRef(liveness.cancel);

  useEffect(() => {
    livenessConsumerRef.current = liveness.consumeMotion;
  }, [liveness.consumeMotion]);
  useEffect(() => {
    cancelLivenessRef.current = liveness.cancel;
  }, [liveness.cancel]);

  const resetComparison = useCallback(() => {
    setDistance(null);
    setIdentityDecision("unavailable");
    comparisonHistoryRef.current = [];
    cancelLivenessRef.current();
  }, []);

  const resetAntiSpoof = useCallback(
    (decision: AntiSpoofDecision = "unavailable") => {
      setMiniFasSample(null);
      setAntiSpoofDecision(decision);
      antiSpoofHistoryRef.current = [];
    },
    [],
  );

  const resetPhoneDetection = useCallback(
    (decision: PhoneDecision = "unavailable") => {
      setPhoneSample(null);
      setPhoneDecision(decision);
      phoneHistoryRef.current = [];
      phoneLatchedRef.current = false;
    },
    [],
  );

  const clearReference = useCallback(() => {
    if (previousObjectUrlRef.current)
      URL.revokeObjectURL(previousObjectUrlRef.current);
    previousObjectUrlRef.current = null;
    setImageUrl(null);
    setFileName("");
    setReferenceState("waiting");
    setReferenceSample(emptyFaceSample);
    resetComparison();
  }, [resetComparison]);

  useEffect(
    () => () => {
      if (previousObjectUrlRef.current)
        URL.revokeObjectURL(previousObjectUrlRef.current);
    },
    [],
  );

  const selectFile = useCallback(
    (file: File | undefined) => {
      if (!file) return;
      if (!["image/jpeg", "image/png", "image/webp"].includes(file.type)) {
        setReferenceState("error");
        return;
      }
      if (previousObjectUrlRef.current)
        URL.revokeObjectURL(previousObjectUrlRef.current);
      const nextUrl = URL.createObjectURL(file);
      previousObjectUrlRef.current = nextUrl;
      setImageUrl(nextUrl);
      setFileName(file.name);
      setReferenceState("processing");
      setReferenceSample(emptyFaceSample);
      resetComparison();
    },
    [resetComparison],
  );

  const analyzeReference = useCallback(async () => {
    if (!imageRef.current || faceApiState !== "ready") return;
    setReferenceState("processing");
    try {
      const sample = await detectFaces(imageRef.current);
      setReferenceSample(sample);
      setReferenceState(
        sample.count === 0
          ? "no-face"
          : sample.count > 1
            ? "multiple"
            : "valid",
      );
    } catch {
      setReferenceState("error");
      setReferenceSample(emptyFaceSample);
    }
  }, [faceApiState]);

  useEffect(() => {
    if (
      faceApiState === "ready" &&
      referenceState === "processing" &&
      imageRef.current?.complete
    ) {
      void analyzeReference();
    }
  }, [analyzeReference, faceApiState, referenceState]);

  useEffect(() => {
    const canvas = referenceCanvasRef.current;
    const image = imageRef.current;
    if (!canvas || !image || !image.naturalWidth) return;
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, canvas.width, canvas.height);
    if (!settings.showBoxes || !referenceSample.box) return;
    const { x, y, width, height } = referenceSample.box;
    context.strokeStyle = "#34d399";
    context.lineWidth = Math.max(3, image.naturalWidth / 250);
    context.strokeRect(x, y, width, height);
  }, [referenceSample, settings.showBoxes]);

  useEffect(() => {
    if (!camera.isActive || faceApiState !== "ready") return;
    let cancelled = false;
    let timer = 0;
    const scan = async () => {
      const video = videoRef.current;
      if (!cancelled && !document.hidden && video?.readyState === 4) {
        try {
          const sample = await detectFaces(video);
          if (cancelled) return;
          setCameraSample(sample);
          if (!settings.antiSpoofEnabled) {
            resetAntiSpoof("disabled");
          } else if (
            sample.count === 1 &&
            sample.box &&
            miniFasState === "ready"
          ) {
            const nextMiniFasSample = await analyzeAntiSpoof(video, sample.box);
            if (cancelled) return;
            setMiniFasSample(nextMiniFasSample);
            const history = [
              ...antiSpoofHistoryRef.current,
              nextMiniFasSample.realProbability,
            ].slice(-3);
            antiSpoofHistoryRef.current = history;
            const realFrames = history.filter(
              (probability) => probability >= settings.antiSpoofThreshold,
            ).length;
            if (history.length < 3) setAntiSpoofDecision("unstable");
            else if (realFrames === 3) setAntiSpoofDecision("real");
            else if (realFrames === 0) setAntiSpoofDecision("spoof");
            else setAntiSpoofDecision("unstable");
          } else {
            resetAntiSpoof();
          }
          if (
            sample.count === 1 &&
            sample.descriptor &&
            referenceSample.descriptor
          ) {
            const nextDistance = await compareDescriptors(
              referenceSample.descriptor,
              sample.descriptor,
            );
            setDistance(nextDistance);
            const history = [
              ...comparisonHistoryRef.current,
              nextDistance <= settings.identityThreshold,
            ].slice(-4);
            comparisonHistoryRef.current = history;
            const matches = history.filter(Boolean).length;
            if (history.length >= 3 && matches >= 3)
              setIdentityDecision("match");
            else if (history.length >= 3 && history.length - matches >= 3)
              setIdentityDecision("no-match");
            else setIdentityDecision("unstable");
          } else resetComparison();
        } catch {
          if (!cancelled) {
            setIdentityDecision("unavailable");
            if (settings.antiSpoofEnabled) resetAntiSpoof();
          }
        }
      }
      if (!cancelled)
        timer = window.setTimeout(scan, liveness.isRunning ? 1350 : 900);
    };
    void scan();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [
    camera.isActive,
    faceApiState,
    liveness.isRunning,
    miniFasState,
    referenceSample.descriptor,
    resetComparison,
    resetAntiSpoof,
    settings.antiSpoofEnabled,
    settings.antiSpoofThreshold,
    settings.identityThreshold,
  ]);

  useEffect(() => {
    if (!camera.isActive || mediaPipeState !== "ready") return;
    if (!liveness.isRunning && !settings.showLandmarks) return;
    let cancelled = false;
    let timer = 0;
    const scanMotion = async () => {
      const video = videoRef.current;
      if (!cancelled && !document.hidden && video?.readyState === 4) {
        try {
          const metrics = await analyzeMotion(video, performance.now());
          if (cancelled) return;
          setMotionMetrics(metrics);
          if (liveness.isRunning) livenessConsumerRef.current(metrics);
        } catch {
          if (liveness.isRunning) cancelLivenessRef.current();
        }
      }
      if (!cancelled)
        timer = window.setTimeout(scanMotion, liveness.isRunning ? 130 : 320);
    };
    void scanMotion();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [
    camera.isActive,
    liveness.isRunning,
    mediaPipeState,
    settings.showLandmarks,
  ]);

  useEffect(() => {
    if (
      !settings.phoneDetectionEnabled ||
      !camera.isActive ||
      objectDetectorState !== "ready"
    )
      return;
    let cancelled = false;
    let timer = 0;
    const scanObjects = async () => {
      const video = videoRef.current;
      if (!cancelled && !document.hidden && video?.readyState === 4) {
        try {
          const sample = await detectPhone(video, performance.now());
          if (cancelled) return;
          setPhoneSample(sample);
          const confidence = sample.confidence ?? 0;
          const lowConfidenceDetection =
            sample.detected &&
            confidence >= settings.phoneConfidenceThreshold;
          const immediateDetection =
            sample.detected && confidence >= PHONE_IMMEDIATE_THRESHOLD;
          const history = [
            ...phoneHistoryRef.current,
            lowConfidenceDetection,
          ].slice(-4);
          phoneHistoryRef.current = history;
          const detectedFrames = history.filter(Boolean).length;
          if (immediateDetection || detectedFrames >= 2) {
            phoneLatchedRef.current = true;
            setPhoneDecision("detected");
          } else if (history.length === 4 && detectedFrames === 0) {
            phoneLatchedRef.current = false;
            setPhoneDecision("clear");
          } else {
            setPhoneDecision(
              phoneLatchedRef.current ? "detected" : "unstable",
            );
          }
        } catch {
          if (!cancelled) resetPhoneDetection();
        }
      }
      if (!cancelled) timer = window.setTimeout(scanObjects, 300);
    };
    void scanObjects();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [
    camera.isActive,
    objectDetectorState,
    resetPhoneDetection,
    settings.phoneConfidenceThreshold,
    settings.phoneDetectionEnabled,
  ]);

  useEffect(() => {
    const canvas = cameraCanvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video?.videoWidth) return;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, canvas.width, canvas.height);
    if (settings.showBoxes && cameraSample.box) {
      const { x, y, width, height } = cameraSample.box;
      context.strokeStyle =
        identityDecision === "match"
          ? "#34d399"
          : identityDecision === "no-match"
            ? "#fb7185"
            : "#fbbf24";
      context.lineWidth = Math.max(3, video.videoWidth / 250);
      context.strokeRect(x, y, width, height);
    }
    if (settings.showBoxes && phoneDecision === "detected" && phoneSample?.box) {
      const { x, y, width, height } = phoneSample.box;
      context.fillStyle = "rgba(244, 63, 94, .14)";
      context.strokeStyle = "#fb7185";
      context.lineWidth = Math.max(4, video.videoWidth / 220);
      context.fillRect(x, y, width, height);
      context.strokeRect(x, y, width, height);
    }
    if (settings.showLandmarks && motionMetrics?.landmarks.length) {
      context.fillStyle = "rgba(103, 232, 249, .8)";
      for (const point of motionMetrics.landmarks) {
        context.beginPath();
        context.arc(
          point.x * canvas.width,
          point.y * canvas.height,
          1.5,
          0,
          Math.PI * 2,
        );
        context.fill();
      }
    }
  }, [
    cameraSample,
    identityDecision,
    motionMetrics,
    phoneDecision,
    phoneSample,
    settings.showBoxes,
    settings.showLandmarks,
  ]);

  const identityLabel = useMemo(() => {
    if (!referenceSample.descriptor) return "No disponible";
    if (cameraSample.count === 0) return "Rostro no detectado";
    if (cameraSample.count > 1) return "Múltiples rostros";
    if (identityDecision === "match") return "Coincide";
    if (identityDecision === "no-match") return "No coincide";
    return identityDecision === "unstable" ? "Estabilizando" : "No disponible";
  }, [cameraSample.count, identityDecision, referenceSample.descriptor]);

  const identityTone =
    identityDecision === "match"
      ? "success"
      : identityDecision === "no-match"
        ? "danger"
        : "pending";
  const livenessLabel =
    liveness.status === "completed"
      ? "Completada"
      : ["failed", "timed-out"].includes(liveness.status)
        ? "Fallida"
        : liveness.isRunning
          ? "En progreso"
          : "No iniciada";
  const livenessTone =
    liveness.status === "completed"
      ? "success"
      : ["failed", "timed-out"].includes(liveness.status)
        ? "danger"
        : "neutral";
  const antiSpoofLabel = !settings.antiSpoofEnabled
    ? "Desactivado"
    : antiSpoofDecision === "real"
      ? "Rostro real"
      : antiSpoofDecision === "spoof"
        ? "Posible suplantación"
        : antiSpoofDecision === "unstable"
          ? "Estabilizando"
          : cameraSample.count === 0
            ? "Rostro no detectado"
            : "No disponible";
  const antiSpoofTone = !settings.antiSpoofEnabled
    ? "neutral"
    : antiSpoofDecision === "real"
      ? "success"
      : antiSpoofDecision === "spoof"
        ? "danger"
        : "pending";
  const phoneLabel = !settings.phoneDetectionEnabled
    ? "Desactivado"
    : phoneDecision === "detected"
      ? "Teléfono detectado"
      : phoneDecision === "clear"
        ? "Sin teléfono"
        : phoneDecision === "unstable"
          ? "Analizando entorno"
          : "No disponible";
  const phoneTone = !settings.phoneDetectionEnabled
    ? "neutral"
    : phoneDecision === "detected"
      ? "danger"
      : phoneDecision === "clear"
        ? "success"
        : "pending";
  const isApproved =
    identityDecision === "match" &&
    (!settings.antiSpoofEnabled || antiSpoofDecision === "real") &&
    (!settings.phoneDetectionEnabled || phoneDecision === "clear") &&
    liveness.status === "completed";
  const isRejected =
    identityDecision === "no-match" ||
    (settings.antiSpoofEnabled && antiSpoofDecision === "spoof") ||
    (settings.phoneDetectionEnabled && phoneDecision === "detected") ||
    ["failed", "timed-out"].includes(liveness.status);
  const canStartLiveness = Boolean(
    settings.livenessEnabled &&
    camera.isActive &&
    referenceSample.descriptor &&
    mediaPipeState === "ready" &&
    (!settings.phoneDetectionEnabled || phoneDecision === "clear") &&
    !liveness.isRunning,
  );

  const toggleAction = (action: LivenessAction, checked: boolean) =>
    setSettings((current) => ({
      ...current,
      enabledActions: checked
        ? Array.from(new Set([...current.enabledActions, action]))
        : current.enabledActions.filter((item) => item !== action),
    }));
  const stopCamera = () => {
    camera.stop();
    setCameraSample(emptyFaceSample);
    setMotionMetrics(null);
    resetAntiSpoof(settings.antiSpoofEnabled ? "unavailable" : "disabled");
    resetPhoneDetection(
      settings.phoneDetectionEnabled ? "unavailable" : "disabled",
    );
    resetComparison();
  };

  return (
    <main className="min-h-screen bg-[#f3f6f4] text-slate-950">
      <header className="relative overflow-hidden bg-[#071d19] text-white">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_80%_0%,rgba(34,197,94,.15),transparent_34%),radial-gradient(circle_at_10%_120%,rgba(6,182,212,.12),transparent_40%)]" />
        <div className="relative mx-auto flex max-w-7xl flex-col gap-6 px-5 py-7 sm:px-8 lg:flex-row lg:items-center lg:justify-between lg:py-9">
          <div className="flex items-center gap-4">
            <div className="grid size-12 place-items-center rounded-2xl border border-emerald-300/15 bg-emerald-300/10 shadow-inner">
              <ScanFace className="size-7 text-emerald-300" />
            </div>
            <div>
              <div className="mb-1 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.18em] text-emerald-300">
                Laboratorio biométrico{" "}
                <span className="size-1 rounded-full bg-emerald-300" /> POC
              </div>
              <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">
                Verificación facial experimental
              </h1>
              <p className="mt-1 max-w-xl text-sm text-slate-300">
                Compara identidad, clasifica presentación, busca teléfonos y
                completa movimientos sin enviar datos fuera del navegador.
              </p>
            </div>
          </div>
          <div className="flex flex-wrap gap-2">
            <ModelBadge label="Face API" state={faceApiState} />
            <ModelBadge label="MiniFASNetV2" state={miniFasState} />
            <ModelBadge label="MediaPipe Face" state={mediaPipeState} />
            <ModelBadge
              label="MediaPipe Object"
              state={objectDetectorState}
            />
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-7xl px-5 py-6 sm:px-8 sm:py-8">
        <Alert className="mb-6 border-amber-200 bg-amber-50 text-amber-950">
          <Info className="size-4" />
          <AlertTitle>Prueba de concepto, no sistema de asistencia</AlertTitle>
          <AlertDescription className="text-amber-800">
            MiniFASNetV2 y la detección de teléfonos son señales experimentales.
            El detector puede omitir teléfonos ocultos o confundirse con otros
            objetos; ninguna señal constituye una prueba de vida certificada.
          </AlertDescription>
        </Alert>
        {modelError ? (
          <Alert variant="destructive" className="mb-6">
            <AlertCircle className="size-4" />
            <AlertTitle>Error al cargar modelos</AlertTitle>
            <AlertDescription className="flex flex-wrap items-center justify-between gap-3">
              {modelError}
              <Button size="sm" variant="outline" onClick={() => void retry()}>
                <RefreshCw /> Reintentar
              </Button>
            </AlertDescription>
          </Alert>
        ) : null}

        <section className="mb-6 grid grid-cols-2 overflow-hidden rounded-2xl border border-slate-200 bg-white shadow-sm lg:grid-cols-5">
          {[
            [
              "1",
              "Referencia",
              referenceState === "valid" ? "Lista" : "Sube una foto",
            ],
            [
              "2",
              "Comparación",
              identityDecision === "match" || identityDecision === "no-match"
                ? "Evaluada"
                : "Activa la cámara",
            ],
            [
              "3",
              "Anti-spoof",
              antiSpoofDecision === "real"
                ? "Rostro real"
                : "Analizando cámara",
            ],
            [
              "4",
              "Entorno",
              phoneDecision === "detected"
                ? "Teléfono detectado"
                : "Buscando teléfonos",
            ],
            [
              "5",
              "Movimientos",
              liveness.status === "completed"
                ? "Completados"
                : "Completa el reto",
            ],
          ].map(([number, title, detail], index) => (
            <div
              key={number}
              className={cn(
                "flex min-w-0 items-center gap-3 px-3 py-4 sm:px-5",
                index % 2 === 1 && "border-l border-slate-200",
                index > 0 && "lg:border-l",
                index > 1 && "border-t border-slate-200 lg:border-t-0",
                index === 4 && "col-span-2 lg:col-span-1",
              )}
            >
              <span className="grid size-7 shrink-0 place-items-center rounded-full bg-emerald-950 text-xs font-bold text-white">
                {number}
              </span>
              <div className="min-w-0">
                <p className="truncate text-sm font-semibold text-slate-900">
                  {title}
                </p>
                <p className="hidden truncate text-xs text-slate-500 sm:block">
                  {detail}
                </p>
              </div>
            </div>
          ))}
        </section>

        <section className="grid gap-6 lg:grid-cols-2">
          <Card className="overflow-hidden border-slate-200 bg-white py-0 shadow-sm">
            <CardHeader className="border-b border-slate-100 px-5 py-5 sm:px-6">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <CardTitle className="flex items-center gap-2 text-lg">
                    <FileImage className="size-5 text-emerald-700" /> Fotografía
                    de referencia
                  </CardTitle>
                  <CardDescription className="mt-1">
                    Usa una imagen frontal, nítida y con una sola persona.
                  </CardDescription>
                </div>
                <Badge
                  variant="outline"
                  className={cn(
                    "rounded-full",
                    referenceState === "valid" &&
                    "border-emerald-200 bg-emerald-50 text-emerald-700",
                    ["no-face", "multiple", "error"].includes(referenceState) &&
                    "border-rose-200 bg-rose-50 text-rose-700",
                  )}
                >
                  {referenceState === "processing" ? (
                    <LoaderCircle className="animate-spin" />
                  ) : referenceState === "valid" ? (
                    <Check />
                  ) : null}
                  {referenceCopy[referenceState]}
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="p-5 sm:p-6">
              <input
                ref={fileInputRef}
                type="file"
                accept="image/jpeg,image/png,image/webp"
                className="hidden"
                onChange={(event) => selectFile(event.target.files?.[0])}
              />
              {imageUrl ? (
                <div>
                  <div className="relative aspect-[4/3] overflow-hidden rounded-2xl bg-slate-950">
                    {/* eslint-disable-next-line @next/next/no-img-element */}
                    <img
                      ref={imageRef}
                      src={imageUrl}
                      alt="Fotografía de referencia seleccionada"
                      className="size-full object-contain"
                      onLoad={() => void analyzeReference()}
                    />
                    <canvas
                      ref={referenceCanvasRef}
                      className="pointer-events-none absolute inset-0 size-full object-contain"
                    />
                    {referenceState === "processing" ? (
                      <div className="absolute inset-0 grid place-items-center bg-slate-950/55 backdrop-blur-sm">
                        <div className="text-center text-white">
                          <LoaderCircle className="mx-auto mb-2 size-7 animate-spin text-emerald-300" />
                          <p className="text-sm font-medium">
                            Generando descriptor…
                          </p>
                        </div>
                      </div>
                    ) : null}
                  </div>
                  <div className="mt-4 flex items-center justify-between gap-3 rounded-xl border border-slate-200 bg-slate-50 p-3">
                    <div className="min-w-0">
                      <p className="truncate text-sm font-medium">{fileName}</p>
                      <p className="text-xs text-slate-500">
                        Se mantiene solo en memoria
                      </p>
                    </div>
                    <div className="flex gap-2">
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => fileInputRef.current?.click()}
                      >
                        <RefreshCw /> Reemplazar
                      </Button>
                      <Button
                        size="icon-sm"
                        variant="ghost"
                        aria-label="Quitar fotografía"
                        onClick={clearReference}
                      >
                        <X />
                      </Button>
                    </div>
                  </div>
                </div>
              ) : (
                <button
                  type="button"
                  onClick={() => fileInputRef.current?.click()}
                  onDragOver={(event) => event.preventDefault()}
                  onDrop={(event) => {
                    event.preventDefault();
                    selectFile(event.dataTransfer.files[0]);
                  }}
                  className="group flex aspect-[4/3] w-full flex-col items-center justify-center rounded-2xl border border-dashed border-slate-300 bg-slate-50 px-6 text-center transition hover:border-emerald-500 hover:bg-emerald-50/50 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-emerald-600"
                >
                  <span className="mb-4 grid size-14 place-items-center rounded-2xl bg-white shadow-sm ring-1 ring-slate-200 transition group-hover:-translate-y-0.5">
                    <Upload className="size-6 text-emerald-700" />
                  </span>
                  <span className="font-semibold">
                    Selecciona o arrastra una fotografía
                  </span>
                  <span className="mt-1 text-sm text-slate-500">
                    JPG, PNG o WebP · un solo rostro
                  </span>
                </button>
              )}
              {["no-face", "multiple", "error"].includes(referenceState) ? (
                <p className="mt-3 flex items-center gap-2 text-sm text-rose-700">
                  <AlertCircle className="size-4" />{" "}
                  {referenceCopy[referenceState]}. Prueba con otra imagen.
                </p>
              ) : null}
            </CardContent>
          </Card>

          <Card className="overflow-hidden border-slate-200 bg-white py-0 shadow-sm">
            <CardHeader className="border-b border-slate-100 px-5 py-5 sm:px-6">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <CardTitle className="flex items-center gap-2 text-lg">
                    <Camera className="size-5 text-emerald-700" /> Cámara en
                    vivo
                  </CardTitle>
                  <CardDescription className="mt-1">
                    Centra tu rostro y procura una iluminación uniforme.
                  </CardDescription>
                </div>
                <Badge
                  variant="outline"
                  className={cn(
                    "rounded-full",
                    camera.isActive
                      ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                      : "text-slate-600",
                  )}
                >
                  <span
                    className={cn(
                      "size-1.5 rounded-full bg-slate-400",
                      camera.isActive && "animate-pulse bg-emerald-500",
                    )}
                  />
                  {camera.isActive ? "Cámara activa" : "Cámara apagada"}
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="p-5 sm:p-6">
              <div className="relative aspect-[4/3] overflow-hidden rounded-2xl bg-[#081713]">
                <video
                  ref={videoRef}
                  muted
                  playsInline
                  className={cn(
                    "size-full -scale-x-100 object-cover",
                    !camera.isActive && "invisible",
                  )}
                />
                <canvas
                  ref={cameraCanvasRef}
                  className="pointer-events-none absolute inset-0 size-full -scale-x-100 object-cover"
                />
                {!camera.isActive ? (
                  <div className="absolute inset-0 grid place-items-center px-6 text-center text-white">
                    <div>
                      <span className="mx-auto mb-4 grid size-16 place-items-center rounded-full bg-white/10 ring-1 ring-white/10">
                        <CameraOff className="size-7 text-emerald-200" />
                      </span>
                      <p className="font-semibold">
                        La cámara está desactivada
                      </p>
                      <p className="mt-1 text-sm text-slate-400">
                        El navegador solicitará permiso al iniciarla.
                      </p>
                    </div>
                  </div>
                ) : null}
                {camera.isActive && cameraSample.count !== 1 ? (
                  <div className="absolute inset-x-4 bottom-4 rounded-xl bg-slate-950/75 px-4 py-3 text-center text-sm font-medium text-white backdrop-blur">
                    {cameraSample.count > 1
                      ? "Hay más de un rostro en el encuadre"
                      : "Coloca un rostro dentro del encuadre"}
                  </div>
                ) : null}
                {camera.isActive && phoneDecision === "detected" ? (
                  <div className="absolute inset-x-4 bottom-4 flex items-center justify-center gap-2 rounded-xl border border-rose-300/30 bg-rose-950/85 px-4 py-3 text-center text-sm font-semibold text-white backdrop-blur">
                    <Smartphone className="size-4 text-rose-200" /> Teléfono
                    detectado en el encuadre
                  </div>
                ) : null}
                {liveness.isRunning && liveness.currentAction ? (
                  <div className="absolute inset-x-4 top-4 rounded-xl border border-emerald-300/25 bg-[#071d19]/90 p-4 text-white shadow-xl backdrop-blur">
                    <div className="flex items-center justify-between gap-3">
                      <p className="text-xs font-semibold uppercase tracking-wider text-emerald-300">
                        Acción {liveness.currentIndex + 1} de{" "}
                        {liveness.sequence.length}
                      </p>
                      <span className="font-mono text-xs">
                        {Math.ceil(liveness.remainingMs / 1000)} s
                      </span>
                    </div>
                    <p className="mt-1 text-lg font-semibold">
                      {actionCopy[liveness.currentAction]}
                    </p>
                    <Progress
                      value={
                        (liveness.remainingMs /
                          (settings.actionTimeoutSeconds * 1000)) *
                        100
                      }
                      className="mt-3 h-1.5 bg-white/15"
                    />
                  </div>
                ) : null}
              </div>
              {camera.error ? (
                <p className="mt-3 flex items-start gap-2 text-sm text-rose-700">
                  <AlertCircle className="mt-0.5 size-4 shrink-0" />{" "}
                  {camera.error}
                </p>
              ) : null}
              <div className="mt-4 flex flex-col gap-3 sm:flex-row">
                {camera.isActive ? (
                  <Button
                    variant="outline"
                    className="h-11 flex-1"
                    onClick={stopCamera}
                  >
                    <CameraOff /> Detener cámara
                  </Button>
                ) : (
                  <Button
                    className="h-11 flex-1 bg-emerald-800 text-white hover:bg-emerald-900"
                    onClick={() => void camera.start(videoRef.current)}
                    disabled={faceApiState !== "ready"}
                  >
                    <Camera /> Iniciar cámara
                  </Button>
                )}
                {camera.devices.length > 1 ? (
                  <Select
                    value={camera.deviceId}
                    onValueChange={(value) => {
                      if (!value) return;
                      camera.setDeviceId(value);
                      if (camera.isActive)
                        void camera.start(videoRef.current, value);
                    }}
                  >
                    <SelectTrigger className="h-11 w-full sm:w-56">
                      <SelectValue placeholder="Elegir cámara" />
                    </SelectTrigger>
                    <SelectContent>
                      {camera.devices.map((device, index) => (
                        <SelectItem
                          key={device.deviceId}
                          value={device.deviceId}
                        >
                          {device.label || `Cámara ${index + 1}`}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                ) : null}
              </div>
            </CardContent>
          </Card>
        </section>

        <section className="mt-6 grid gap-6 xl:grid-cols-2">
          <Card className="border-slate-200 bg-white shadow-sm">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-lg">
                <Activity className="size-5 text-emerald-700" /> Comparación de
                identidad
              </CardTitle>
              <CardDescription>
                La decisión se estabiliza con al menos tres lecturas
                concordantes.
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div
                className={cn(
                  "mb-4 flex items-center gap-4 rounded-2xl border p-4",
                  identityDecision === "match"
                    ? "border-emerald-200 bg-emerald-50"
                    : identityDecision === "no-match"
                      ? "border-rose-200 bg-rose-50"
                      : "border-slate-200 bg-slate-50",
                )}
              >
                <span
                  className={cn(
                    "grid size-11 shrink-0 place-items-center rounded-full bg-white shadow-sm",
                    identityDecision === "match"
                      ? "text-emerald-700"
                      : identityDecision === "no-match"
                        ? "text-rose-700"
                        : "text-slate-500",
                  )}
                >
                  {identityDecision === "match" ? (
                    <ShieldCheck />
                  ) : identityDecision === "no-match" ? (
                    <ShieldX />
                  ) : (
                    <CircleUserRound />
                  )}
                </span>
                <div>
                  <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                    Resultado actual
                  </p>
                  <p className="text-xl font-semibold">{identityLabel}</p>
                </div>
              </div>
              <div className="grid grid-cols-3 gap-2 sm:gap-3">
                <Metric
                  label="Distancia"
                  value={distance === null ? "—" : distance.toFixed(4)}
                />
                <Metric
                  label="Umbral"
                  value={settings.identityThreshold.toFixed(2)}
                />
                <Metric
                  label="Inferencia"
                  value={
                    cameraSample.inferenceMs
                      ? `${Math.round(cameraSample.inferenceMs)} ms`
                      : "—"
                  }
                />
              </div>
              <p className="mt-3 text-xs leading-relaxed text-slate-500">
                Una distancia menor indica descriptores más similares. El umbral
                de 0.50 es experimental: no es probabilidad ni porcentaje de
                identidad.
              </p>
            </CardContent>
          </Card>

          <Card className="border-slate-200 bg-white shadow-sm">
            <CardHeader>
              <div className="flex items-start justify-between gap-3">
                <div>
                  <CardTitle className="flex items-center gap-2 text-lg">
                    <ShieldCheck className="size-5 text-emerald-700" />
                    Anti-spoof pasivo
                  </CardTitle>
                  <CardDescription className="mt-1">
                    MiniFASNetV2 clasifica un recorte BGR 80×80 como real o
                    spoof.
                  </CardDescription>
                </div>
                <Switch
                  checked={settings.antiSpoofEnabled}
                  onCheckedChange={(checked) => {
                    setSettings((current) => ({
                      ...current,
                      antiSpoofEnabled: checked,
                    }));
                    resetAntiSpoof(checked ? "unavailable" : "disabled");
                  }}
                  aria-label="Activar anti-spoof pasivo"
                />
              </div>
            </CardHeader>
            <CardContent>
              <div
                className={cn(
                  "mb-4 flex items-center gap-4 rounded-2xl border p-4",
                  antiSpoofDecision === "real"
                    ? "border-emerald-200 bg-emerald-50"
                    : antiSpoofDecision === "spoof"
                      ? "border-rose-200 bg-rose-50"
                      : "border-slate-200 bg-slate-50",
                )}
              >
                <span
                  className={cn(
                    "grid size-11 shrink-0 place-items-center rounded-full bg-white shadow-sm",
                    antiSpoofDecision === "real"
                      ? "text-emerald-700"
                      : antiSpoofDecision === "spoof"
                        ? "text-rose-700"
                        : "text-slate-500",
                  )}
                >
                  {antiSpoofDecision === "real" ? (
                    <ShieldCheck />
                  ) : antiSpoofDecision === "spoof" ? (
                    <ShieldX />
                  ) : (
                    <ScanFace />
                  )}
                </span>
                <div>
                  <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                    Resultado actual
                  </p>
                  <p className="text-xl font-semibold">{antiSpoofLabel}</p>
                </div>
              </div>
              <div className="grid grid-cols-3 gap-2">
                <Metric
                  label="Puntaje real"
                  value={
                    miniFasSample
                      ? `${Math.round(miniFasSample.realProbability * 100)}%`
                      : "—"
                  }
                />
                <Metric
                  label="Umbral"
                  value={`${Math.round(settings.antiSpoofThreshold * 100)}%`}
                />
                <Metric
                  label="Inferencia"
                  value={
                    miniFasSample
                      ? `${Math.round(miniFasSample.inferenceMs)} ms`
                      : "—"
                  }
                />
              </div>
              <p className="mt-3 text-xs leading-relaxed text-slate-500">
                El puntaje no es una probabilidad calibrada ni una garantía de
                vida. La decisión exige tres cuadros consecutivos concordantes.
              </p>
            </CardContent>
          </Card>

          <Card className="border-slate-200 bg-white shadow-sm">
            <CardHeader>
              <div className="flex items-start justify-between gap-3">
                <div>
                  <CardTitle className="flex items-center gap-2 text-lg">
                    <Smartphone className="size-5 text-emerald-700" /> Control
                    del entorno
                  </CardTitle>
                  <CardDescription className="mt-1">
                    MediaPipe Object Detector busca teléfonos en la cámara.
                  </CardDescription>
                </div>
                <Switch
                  checked={settings.phoneDetectionEnabled}
                  onCheckedChange={(checked) => {
                    setSettings((current) => ({
                      ...current,
                      phoneDetectionEnabled: checked,
                    }));
                    resetPhoneDetection(checked ? "unavailable" : "disabled");
                  }}
                  aria-label="Activar detección de teléfonos"
                />
              </div>
            </CardHeader>
            <CardContent>
              <div
                className={cn(
                  "mb-4 flex items-center gap-4 rounded-2xl border p-4",
                  phoneDecision === "clear"
                    ? "border-emerald-200 bg-emerald-50"
                    : phoneDecision === "detected"
                      ? "border-rose-200 bg-rose-50"
                      : "border-slate-200 bg-slate-50",
                )}
              >
                <span
                  className={cn(
                    "grid size-11 shrink-0 place-items-center rounded-full bg-white shadow-sm",
                    phoneDecision === "clear"
                      ? "text-emerald-700"
                      : phoneDecision === "detected"
                        ? "text-rose-700"
                        : "text-slate-500",
                  )}
                >
                  {phoneDecision === "clear" ? (
                    <Check />
                  ) : phoneDecision === "detected" ? (
                    <Smartphone />
                  ) : (
                    <ScanFace />
                  )}
                </span>
                <div>
                  <p className="text-xs font-semibold uppercase tracking-wider text-slate-500">
                    Resultado actual
                  </p>
                  <p className="text-xl font-semibold">{phoneLabel}</p>
                </div>
              </div>
              <div className="grid grid-cols-3 gap-2">
                <Metric
                  label="Confianza"
                  value={
                    phoneSample?.confidence === null || !phoneSample
                      ? "—"
                      : `${Math.round(phoneSample.confidence * 100)}%`
                  }
                />
                <Metric
                  label="Umbral bajo"
                  value={`${Math.round(settings.phoneConfidenceThreshold * 100)}%`}
                />
                <Metric
                  label="Inferencia"
                  value={
                    phoneSample
                      ? `${Math.round(phoneSample.inferenceMs)} ms`
                      : "—"
                  }
                />
              </div>
              <p className="mt-3 text-xs leading-relaxed text-slate-500">
                Bloquea al instante desde 45% o con dos detecciones de cuatro
                desde el umbral bajo. No demuestra que el teléfono esté grabando.
              </p>
            </CardContent>
          </Card>

          <Card className="border-slate-200 bg-white shadow-sm">
            <CardHeader>
              <div className="flex items-start justify-between gap-3">
                <div>
                  <CardTitle className="flex items-center gap-2 text-lg">
                    <Eye className="size-5 text-emerald-700" /> Verificación
                    activa
                  </CardTitle>
                  <CardDescription className="mt-1">
                    Acciones aleatorias detectadas con landmarks y blendshapes.
                  </CardDescription>
                </div>
                <Switch
                  checked={settings.livenessEnabled}
                  onCheckedChange={(checked) => {
                    setSettings((current) => ({
                      ...current,
                      livenessEnabled: checked,
                    }));
                    if (!checked) liveness.cancel();
                  }}
                  aria-label="Activar verificación activa"
                />
              </div>
            </CardHeader>
            <CardContent>
              {!settings.livenessEnabled ? (
                <div className="rounded-2xl border border-slate-200 bg-slate-50 p-5 text-center">
                  <Eye className="mx-auto mb-2 size-7 text-slate-400" />
                  <p className="font-semibold">Solo comparación facial</p>
                  <p className="mt-1 text-sm text-slate-500">
                    No se emitirá una aprobación conjunta.
                  </p>
                </div>
              ) : (
                <div>
                  <div className="mb-4 flex flex-wrap gap-2">
                    {liveness.sequence.length ? (
                      liveness.sequence.map((action, index) => (
                        <div
                          key={`${action}-${index}`}
                          className={cn(
                            "flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium",
                            index < liveness.currentIndex
                              ? "border-emerald-200 bg-emerald-50 text-emerald-700"
                              : index === liveness.currentIndex &&
                                liveness.isRunning
                                ? "border-amber-200 bg-amber-50 text-amber-800"
                                : "border-slate-200 text-slate-500",
                          )}
                        >
                          {index < liveness.currentIndex ? (
                            <Check className="size-3.5" />
                          ) : (
                            <span className="size-4 rounded-full border text-center text-[10px] leading-[14px]">
                              {index + 1}
                            </span>
                          )}
                          {actionCopy[action]}
                        </div>
                      ))
                    ) : (
                      <p className="text-sm text-slate-500">
                        La secuencia se genera al iniciar cada intento.
                      </p>
                    )}
                  </div>
                  {liveness.failureReason ? (
                    <p className="mb-3 flex items-start gap-2 rounded-xl bg-rose-50 p-3 text-sm text-rose-700">
                      <AlertCircle className="mt-0.5 size-4 shrink-0" />{" "}
                      {liveness.failureReason}
                    </p>
                  ) : null}
                  {liveness.status === "completed" ? (
                    <p className="mb-3 flex items-center gap-2 rounded-xl bg-emerald-50 p-3 text-sm font-medium text-emerald-700">
                      <Check className="size-4" /> Secuencia completada.
                    </p>
                  ) : null}
                  <div className="flex gap-2">
                    <Button
                      className="h-11 flex-1 bg-emerald-800 text-white hover:bg-emerald-900"
                      disabled={!canStartLiveness}
                      onClick={liveness.start}
                    >
                      {liveness.isRunning ? (
                        <LoaderCircle className="animate-spin" />
                      ) : (
                        <Eye />
                      )}{" "}
                      {liveness.isRunning
                        ? "Verificando…"
                        : liveness.status === "completed"
                          ? "Repetir verificación"
                          : "Iniciar verificación"}
                    </Button>
                    {liveness.isRunning ? (
                      <Button
                        variant="outline"
                        className="h-11"
                        onClick={liveness.cancel}
                      >
                        Cancelar
                      </Button>
                    ) : null}
                  </div>
                  {!canStartLiveness && !liveness.isRunning ? (
                    <p className="mt-2 text-center text-xs text-slate-500">
                      Necesitas una referencia válida, cámara activa, MediaPipe
                      listo y ningún teléfono detectado.
                    </p>
                  ) : null}
                </div>
              )}
            </CardContent>
          </Card>
        </section>

        <Card className="mt-6 border-slate-200 bg-white shadow-sm">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-lg">
              <ShieldCheck className="size-5 text-emerald-700" /> Resumen de la
              POC
            </CardTitle>
            <CardDescription>
              Cuatro señales independientes y una decisión conjunta; no se
              registra asistencia.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid xl:grid-cols-5 xl:divide-x xl:divide-slate-200">
              <div className="xl:pr-5">
                <ResultRow
                  label="Identidad facial"
                  value={identityLabel}
                  tone={identityTone}
                  detail={
                    distance === null
                      ? "Sin comparación estable"
                      : `Distancia ${distance.toFixed(4)}`
                  }
                />
              </div>
              <div className="border-t border-slate-200 xl:border-t-0 xl:px-5">
                <ResultRow
                  label="Anti-spoof pasivo"
                  value={antiSpoofLabel}
                  tone={antiSpoofTone}
                  detail={
                    miniFasSample
                      ? `Real ${Math.round(miniFasSample.realProbability * 100)}%`
                      : "MiniFASNetV2"
                  }
                />
              </div>
              <div className="border-t border-slate-200 xl:border-t-0 xl:px-5">
                <ResultRow
                  label="Teléfono en cámara"
                  value={phoneLabel}
                  tone={phoneTone}
                  detail={
                    phoneSample?.confidence == null
                      ? "EfficientDet-Lite0"
                      : `Confianza ${Math.round(phoneSample.confidence * 100)}%`
                  }
                />
              </div>
              <div className="border-t border-slate-200 xl:border-t-0 xl:px-5">
                <ResultRow
                  label="Verificación activa"
                  value={
                    settings.livenessEnabled ? livenessLabel : "Desactivada"
                  }
                  tone={settings.livenessEnabled ? livenessTone : "neutral"}
                  detail={
                    settings.livenessEnabled
                      ? "Movimiento experimental"
                      : "Solo comparación facial"
                  }
                />
              </div>
              <div className="border-t border-slate-200 xl:border-t-0 xl:pl-5">
                <ResultRow
                  label="Resultado general"
                  value={
                    !settings.livenessEnabled
                      ? "Solo comparación facial"
                      : isApproved
                        ? "Aprobado"
                        : isRejected
                          ? "Rechazado"
                          : "Pendiente"
                  }
                  tone={
                    !settings.livenessEnabled
                      ? "neutral"
                      : isApproved
                        ? "success"
                        : isRejected
                          ? "danger"
                          : "pending"
                  }
                  detail={
                    settings.livenessEnabled
                      ? "Requiere identidad, anti-spoof, entorno y movimientos"
                      : "Sin aprobación conjunta"
                  }
                />
              </div>
            </div>
          </CardContent>
        </Card>

        <Card className="mt-6 border-slate-200 bg-white shadow-sm">
          <Accordion>
            <AccordionItem value="advanced" className="border-0 px-5 sm:px-6">
              <AccordionTrigger className="py-5 hover:no-underline">
                <span className="flex items-center gap-2 text-base">
                  <Settings2 className="size-5 text-emerald-700" /> Panel de
                  pruebas avanzadas
                </span>
              </AccordionTrigger>
              <AccordionContent className="pb-6">
                <Separator className="mb-6" />
                <div className="grid gap-7 lg:grid-cols-2">
                  <div className="space-y-6">
                    <div>
                      <div className="mb-3 flex items-center justify-between">
                        <Label>Umbral de distancia facial</Label>
                        <span className="font-mono text-sm font-semibold">
                          {settings.identityThreshold.toFixed(2)}
                        </span>
                      </div>
                      <Slider
                        min={0.3}
                        max={0.9}
                        step={0.01}
                        value={[settings.identityThreshold]}
                        onValueChange={(value) =>
                          setSettings((current) => ({
                            ...current,
                            identityThreshold: Array.isArray(value)
                              ? value[0]
                              : value,
                          }))
                        }
                      />
                      <div className="mt-2 flex justify-between text-xs text-slate-500">
                        <span>Más estricto · 0.30</span>
                        <span>Más permisivo · 0.90</span>
                      </div>
                    </div>
                    <div>
                      <div className="mb-3 flex items-center justify-between">
                        <Label>Tiempo máximo por acción</Label>
                        <span className="font-mono text-sm font-semibold">
                          {settings.actionTimeoutSeconds} s
                        </span>
                      </div>
                      <Slider
                        min={4}
                        max={15}
                        step={1}
                        value={[settings.actionTimeoutSeconds]}
                        onValueChange={(value) =>
                          setSettings((current) => ({
                            ...current,
                            actionTimeoutSeconds: Array.isArray(value)
                              ? value[0]
                              : value,
                          }))
                        }
                      />
                    </div>
                    <div>
                      <div className="mb-3 flex items-center justify-between">
                        <Label>Umbral anti-spoof (real)</Label>
                        <span className="font-mono text-sm font-semibold">
                          {settings.antiSpoofThreshold.toFixed(2)}
                        </span>
                      </div>
                      <Slider
                        min={0.3}
                        max={0.9}
                        step={0.01}
                        value={[settings.antiSpoofThreshold]}
                        onValueChange={(value) =>
                          setSettings((current) => ({
                            ...current,
                            antiSpoofThreshold: Array.isArray(value)
                              ? value[0]
                              : value,
                          }))
                        }
                      />
                      <div className="mt-2 flex justify-between text-xs text-slate-500">
                        <span>Más permisivo · 0.30</span>
                        <span>Más estricto · 0.90</span>
                      </div>
                    </div>
                    <div>
                      <div className="mb-3 flex items-center justify-between">
                        <Label>Umbral de parpadeo</Label>
                        <span className="font-mono text-sm font-semibold">
                          {settings.blinkThreshold.toFixed(2)}
                        </span>
                      </div>
                      <Slider
                        min={0.25}
                        max={0.85}
                        step={0.01}
                        value={[settings.blinkThreshold]}
                        onValueChange={(value) =>
                          setSettings((current) => ({
                            ...current,
                            blinkThreshold: Array.isArray(value)
                              ? value[0]
                              : value,
                          }))
                        }
                      />
                    </div>
                    <div>
                      <div className="mb-3 flex items-center justify-between">
                        <Label>Umbral sensible para teléfono</Label>
                        <span className="font-mono text-sm font-semibold">
                          {settings.phoneConfidenceThreshold.toFixed(2)}
                        </span>
                      </div>
                      <Slider
                        min={0.1}
                        max={0.8}
                        step={0.01}
                        value={[settings.phoneConfidenceThreshold]}
                        onValueChange={(value) =>
                          setSettings((current) => ({
                            ...current,
                            phoneConfidenceThreshold: Array.isArray(value)
                              ? value[0]
                              : value,
                          }))
                        }
                      />
                      <div className="mt-2 flex justify-between text-xs text-slate-500">
                        <span>Más sensible · 0.10</span>
                        <span>Menos falsos positivos · 0.80</span>
                      </div>
                    </div>
                    <div>
                      <div className="mb-3 flex items-center justify-between">
                        <Label>Umbral de giro</Label>
                        <span className="font-mono text-sm font-semibold">
                          {settings.turnThreshold.toFixed(2)}
                        </span>
                      </div>
                      <Slider
                        min={0.1}
                        max={0.5}
                        step={0.01}
                        value={[settings.turnThreshold]}
                        onValueChange={(value) =>
                          setSettings((current) => ({
                            ...current,
                            turnThreshold: Array.isArray(value)
                              ? value[0]
                              : value,
                          }))
                        }
                      />
                    </div>
                  </div>
                  <div>
                    <Label className="mb-3 block">Desafíos disponibles</Label>
                    <div className="grid gap-2 sm:grid-cols-2">
                      {(Object.keys(actionCopy) as LivenessAction[]).map(
                        (action) => (
                          <label
                            key={action}
                            className="flex cursor-pointer items-center gap-3 rounded-xl border border-slate-200 p-3 text-sm"
                          >
                            <Checkbox
                              checked={settings.enabledActions.includes(action)}
                              onCheckedChange={(checked) =>
                                toggleAction(action, checked)
                              }
                            />{" "}
                            {actionCopy[action]}
                          </label>
                        ),
                      )}
                    </div>
                    <Separator className="my-6" />
                    <div className="space-y-4">
                      <div className="flex items-center justify-between gap-4">
                        <div>
                          <Label>Mostrar recuadros</Label>
                          <p className="text-xs text-slate-500">
                            Superpone la caja de detección.
                          </p>
                        </div>
                        <Switch
                          checked={settings.showBoxes}
                          onCheckedChange={(checked) =>
                            setSettings((current) => ({
                              ...current,
                              showBoxes: checked,
                            }))
                          }
                        />
                      </div>
                      <div className="flex items-center justify-between gap-4">
                        <div>
                          <Label>Mostrar landmarks</Label>
                          <p className="text-xs text-slate-500">
                            Aumenta el consumo al mantener MediaPipe activo.
                          </p>
                        </div>
                        <Switch
                          checked={settings.showLandmarks}
                          onCheckedChange={(checked) =>
                            setSettings((current) => ({
                              ...current,
                              showLandmarks: checked,
                            }))
                          }
                        />
                      </div>
                    </div>
                    <Separator className="my-6" />
                    <div className="grid grid-cols-2 gap-3">
                      <Metric
                        label="Face API"
                        value={
                          faceApiState === "ready"
                            ? "Modelo listo"
                            : faceApiState
                        }
                      />
                      <Metric
                        label="MediaPipe Face"
                        value={
                          mediaPipeState === "ready"
                            ? "Modelo listo"
                            : mediaPipeState
                        }
                      />
                      <Metric
                        label="MiniFASNetV2"
                        value={
                          miniFasState === "ready"
                            ? "Modelo listo"
                            : miniFasState
                        }
                      />
                      <Metric
                        label="MediaPipe Object"
                        value={
                          objectDetectorState === "ready"
                            ? "Modelo listo"
                            : objectDetectorState
                        }
                      />
                      <Metric
                        label="Detección facial"
                        value={
                          cameraSample.inferenceMs
                            ? `${Math.round(cameraSample.inferenceMs)} ms`
                            : "—"
                        }
                      />
                      <Metric
                        label="Landmarks"
                        value={
                          motionMetrics?.inferenceMs
                            ? `${Math.round(motionMetrics.inferenceMs)} ms`
                            : "—"
                        }
                      />
                      <Metric
                        label="Anti-spoof"
                        value={
                          miniFasSample
                            ? `${Math.round(miniFasSample.inferenceMs)} ms`
                            : "—"
                        }
                      />
                      <Metric
                        label="Objetos"
                        value={
                          phoneSample
                            ? `${Math.round(phoneSample.inferenceMs)} ms`
                            : "—"
                        }
                      />
                    </div>
                  </div>
                </div>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </Card>

        <footer className="mt-7 flex flex-col gap-3 border-t border-slate-200 py-6 text-xs leading-relaxed text-slate-500 sm:flex-row sm:items-center sm:justify-between">
          <p className="max-w-2xl">
            Procesamiento local en memoria. Las imágenes, descriptores y video
            no se persisten ni se envían a un servidor.
          </p>
          <p className="flex items-center gap-1.5">
            <Gauge className="size-4" /> Face API + MiniFASNetV2 + MediaPipe ·
            inferencia local espaciada
          </p>
        </footer>
      </div>
    </main>
  );
}
