"use client";

import React, {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";
import { useStream } from "./StreamProvider";

type ModelInfo = {
  id: string;
  label: string;
  baseHttp: string;
  wsUrl: string;
};

type ModelContextType = {
  models: ModelInfo[];
  selectedModelId: string;
  selectModel: (id: string) => void;
  enabled: boolean;
  setEnabled: (v: boolean) => void;
  connect: () => Promise<void>;
  disconnect: () => void;
  status: "idle" | "connecting" | "connected" | "error" | "disconnecting";
  error: string | null;
  resolution: { width: number; height: number } | null;
  latestFrameUrl: string | null;
};

const ModelContext = createContext<ModelContextType | undefined>(undefined);

const DEFAULT_MODELS: ModelInfo[] = [
  {
    id: "agin_local",
    label: "Agin",
    baseHttp: "http://127.0.0.1:8000/services/agin",
    wsUrl: "ws://127.0.0.1:8000/services/agin/ws/stream",
  },
];

export const ModelProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const { stream, flipH, flipV, rotate90 } = useStream();
  const [models] = useState<ModelInfo[]>(DEFAULT_MODELS);
  const [selectedModelId, setSelectedModelId] = useState<string>(
    DEFAULT_MODELS[0].id,
  );
  const [enabled, setEnabled] = useState<boolean>(true);
  const [status, setStatus] = useState<ModelContextType["status"]>("idle");
  const [error, setError] = useState<string | null>(null);
  const [resolution, setResolution] = useState<{
    width: number;
    height: number;
  } | null>(null);
  const [latestFrameUrl, setLatestFrameUrl] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const captureLoopRef = useRef<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const lastObjectUrlRef = useRef<string | null>(null);

  const selectedModel = models.find((m) => m.id === selectedModelId)!;

  const flipHRef = useRef<boolean>(flipH);
  const flipVRef = useRef<boolean>(flipV);
  const rotateRef = useRef<boolean>(rotate90);
  useEffect(() => {
    flipHRef.current = flipH;
  }, [flipH]);
  useEffect(() => {
    flipVRef.current = flipV;
  }, [flipV]);
  useEffect(() => {
    rotateRef.current = rotate90;
  }, [rotate90]);

  const revokeLastObjectUrl = () => {
    if (lastObjectUrlRef.current) {
      try {
        URL.revokeObjectURL(lastObjectUrlRef.current);
      } catch {}
      lastObjectUrlRef.current = null;
      setLatestFrameUrl(null);
    }
  };

  const fetchResolution = async () => {
    try {
      const res = await fetch(`${selectedModel.baseHttp}/resolution`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      if (
        json &&
        typeof json.width === "number" &&
        typeof json.height === "number"
      ) {
        setResolution({ width: json.width, height: json.height });
        return { width: json.width, height: json.height };
      } else {
        throw new Error("Invalid resolution response");
      }
    } catch (err: any) {
      console.error("fetchResolution error:", err);
      setResolution(null);
      setError("Failed to fetch model resolution");
      throw err;
    }
  };

  const sendFrame = async () => {
    const ws = wsRef.current;
    const streamLocal = stream;
    if (!ws || ws.readyState !== WebSocket.OPEN || !streamLocal) return;

    const videoTracks = streamLocal.getVideoTracks();
    if (videoTracks.length === 0) return;

    let videoEl = (canvasRef.current as any)?._videoEl;
    if (!videoEl) {
      videoEl = document.createElement("video");
      videoEl.playsInline = true;
      videoEl.muted = true;
      videoEl.autoplay = false;
      (canvasRef.current as any)._videoEl = videoEl;
    }

    if (videoEl.srcObject !== streamLocal) {
      videoEl.srcObject = streamLocal;
      try {
        await videoEl.play();
      } catch {}
    }

    const settings = videoTracks[0].getSettings();
    const camWidth = settings.width!;
    const camHeight = settings.height!;

    const canvas = canvasRef.current!;
    const curRotate = rotateRef.current;

    if (curRotate) {
      if (canvas.width !== camHeight || canvas.height !== camWidth) {
        canvas.width = camHeight;
        canvas.height = camWidth;
      }
    } else {
      if (canvas.width !== camWidth || canvas.height !== camHeight) {
        canvas.width = camWidth;
        canvas.height = camHeight;
      }
    }

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const curFlipH = flipHRef.current;
    const curFlipV = flipVRef.current;

    ctx.save();
    if (curRotate) {
      ctx.translate(canvas.width / 2, canvas.height / 2);
      const scaleX = curFlipH ? -1 : 1;
      const scaleY = curFlipV ? -1 : 1;
      ctx.scale(scaleX, scaleY);
      ctx.rotate(Math.PI / 2);
      try {
        ctx.drawImage(
          videoEl,
          -camWidth / 2,
          -camHeight / 2,
          camWidth,
          camHeight,
        );
      } catch {
        ctx.restore();
        return;
      }
    } else {
      if (curFlipH && curFlipV) {
        ctx.translate(canvas.width, canvas.height);
        ctx.scale(-1, -1);
      } else if (curFlipH) {
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
      } else if (curFlipV) {
        ctx.translate(0, canvas.height);
        ctx.scale(1, -1);
      }
      try {
        ctx.drawImage(videoEl, 0, 0, camWidth, camHeight);
      } catch {
        ctx.restore();
        return;
      }
    }

    await new Promise<void>((resolve) => {
      canvas.toBlob(
        (blob) => {
          if (blob) {
            try {
              ws.send(blob);
            } catch {}
          }
          resolve();
        },
        "image/jpeg",
        0.9,
      );
    });
    ctx.restore();
  };

  const startCaptureLoop = () => {
    let lastSent = 0;
    const TARGET_FPS = 25;
    const interval = 1000 / TARGET_FPS;

    const step = (ts: number) => {
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;
      if (!stream) return;

      if (ts - lastSent >= interval) {
        lastSent = ts;
        sendFrame().catch(() => {});
      }
      captureLoopRef.current = requestAnimationFrame(step);
    };
    captureLoopRef.current = requestAnimationFrame(step);
  };

  const stopCaptureLoop = () => {
    if (captureLoopRef.current) {
      cancelAnimationFrame(captureLoopRef.current);
      captureLoopRef.current = null;
    }
  };

  const handleWsMessage = (ev: MessageEvent) => {
    const data = ev.data;
    let blobPromise: Promise<Blob>;
    if (data instanceof Blob) {
      blobPromise = Promise.resolve(data);
    } else if (data instanceof ArrayBuffer) {
      blobPromise = Promise.resolve(new Blob([data], { type: "image/jpeg" }));
    } else {
      return;
    }

    blobPromise.then((blob) => {
      try {
        revokeLastObjectUrl();
        const obj = URL.createObjectURL(blob);
        lastObjectUrlRef.current = obj;
        setLatestFrameUrl(obj);
      } catch (e) {
        console.error("Failed to create object URL from blob", e);
      }
    });
  };

  const connect = async () => {
    if (status === "connecting" || status === "connected") return;
    setError(null);
    setStatus("connecting");

    try {
      const res = await fetchResolution().catch((err) => {
        throw err;
      });

      if (!canvasRef.current) {
        const c = document.createElement("canvas");
        canvasRef.current = c;
      }

      const ws = new WebSocket(selectedModel.wsUrl);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onopen = () => {
        setStatus("connected");
        setError(null);
        if (stream) startCaptureLoop();
      };

      ws.onmessage = handleWsMessage;

      ws.onerror = (ev) => {
        console.error("WebSocket error", ev);
        setError("WebSocket error");
        setStatus("error");
      };

      ws.onclose = () => {
        setStatus((prev) => (prev === "disconnecting" ? "idle" : "idle"));
        stopCaptureLoop();
        wsRef.current = null;
      };
    } catch (err: any) {
      console.error("connect failed", err);
      setStatus("error");
      setError(
        typeof err === "string"
          ? err
          : (err?.message ?? "Failed to connect to model"),
      );
      stopCaptureLoop();
      if (wsRef.current) {
        try {
          wsRef.current.close();
        } catch {}
        wsRef.current = null;
      }
    }
  };

  const disconnect = () => {
    setStatus("disconnecting");
    stopCaptureLoop();
    try {
      if (wsRef.current) {
        wsRef.current.onopen = null;
        wsRef.current.onmessage = null;
        wsRef.current.onerror = null;
        wsRef.current.onclose = null;
        wsRef.current.close();
        wsRef.current = null;
      }
    } catch (e) {
      console.error("Error closing ws:", e);
    } finally {
      setStatus("idle");
      setError(null);
      revokeLastObjectUrl();
    }
  };

  useEffect(() => {
    if (!enabled) return;
    if (stream && status !== "connected" && status !== "connecting") {
      connect().catch(() => {});
    }
    if (!stream) {
      stopCaptureLoop();
    } else {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        if (!captureLoopRef.current) startCaptureLoop();
      }
    }
  }, [stream, enabled, selectedModelId]);

  useEffect(() => {
    if (enabled) {
      if (status !== "connected" && status !== "connecting") {
        connect().catch(() => {});
      }
    } else {
      disconnect();
    }
  }, [enabled, selectedModelId]);

  useEffect(() => {
    return () => {
      disconnect();
    };
  }, []);

  const value: ModelContextType = {
    models,
    selectedModelId,
    selectModel: (id: string) => setSelectedModelId(id),
    enabled,
    setEnabled,
    connect: async () => connect(),
    disconnect,
    status,
    error,
    resolution,
    latestFrameUrl,
  };

  return (
    <ModelContext.Provider value={value}>{children}</ModelContext.Provider>
  );
};

export function useModel() {
  const ctx = useContext(ModelContext);
  if (!ctx) throw new Error("useModel must be used inside ModelProvider");
  return ctx;
}
