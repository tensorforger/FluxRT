"use client";

import React, {
  createContext,
  useContext,
  useEffect,
  useRef,
  useState,
} from "react";

type StreamContextType = {
  stream: MediaStream | null;
  error: string | null;
  isOpening: boolean;
  devices: MediaDeviceInfo[];
  refreshDevices: () => Promise<void>;
  openDevice: (deviceId: string) => Promise<void>;
  closeStream: () => void;

  flipH: boolean;
  flipV: boolean;
  rotate90: boolean;
  setFlipH: (v: boolean) => void;
  setFlipV: (v: boolean) => void;
  setRotate: (v: boolean) => void;
};

const StreamContext = createContext<StreamContextType | undefined>(undefined);

const FLIP_STORAGE_KEY = "stream:flip";
const LAST_DEVICE_KEY = "lastCameraId";

export const StreamProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const [stream, setStream] = useState<MediaStream | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isOpening, setIsOpening] = useState(false);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const currentStreamRef = useRef<MediaStream | null>(null);

  const [flipH, setFlipHState] = useState<boolean>(false);
  const [flipV, setFlipVState] = useState<boolean>(false);
  const [rotate90, setRotate90State] = useState<boolean>(false);

  useEffect(() => {
    try {
      const raw = localStorage.getItem(FLIP_STORAGE_KEY);
      if (raw) {
        const parsed = JSON.parse(raw);
        setFlipHState(Boolean(parsed.flipH));
        setFlipVState(Boolean(parsed.flipV));
        setRotate90State(Boolean(parsed.rotate90));
      }
    } catch {}
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(
        FLIP_STORAGE_KEY,
        JSON.stringify({ flipH, flipV, rotate90 }),
      );
    } catch {}
  }, [flipH, flipV, rotate90]);

  const refreshDevices = async () => {
    if (!("mediaDevices" in navigator)) {
      setDevices([]);
      return;
    }
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true,
      });

      stream.getTracks().forEach((track) => track.stop());
      const list = await navigator.mediaDevices.enumerateDevices();
      console.log(list);
      setDevices(list.filter((d) => d.kind === "videoinput"));
    } catch (e) {
      console.error("enumerateDevices failed", e);
      setDevices([]);
    }
  };

  useEffect(() => {
    refreshDevices();
    const handler = () => refreshDevices();

    if (
      "mediaDevices" in navigator &&
      "addEventListener" in navigator.mediaDevices
    ) {
      navigator.mediaDevices.addEventListener("devicechange", handler);
      return () =>
        navigator.mediaDevices.removeEventListener("devicechange", handler);
    }
  }, []);

  const closeStream = () => {
    if (currentStreamRef.current) {
      currentStreamRef.current.getTracks().forEach((t) => t.stop());
      currentStreamRef.current = null;
      setStream(null);
    }
    setError(null);
  };

  const openDevice = async (deviceId: string) => {
    if (!("mediaDevices" in navigator)) {
      setError("MediaDevices API not available");
      return;
    }

    setIsOpening(true);
    setError(null);

    try {
      closeStream();

      const s = await navigator.mediaDevices.getUserMedia({
        video: {
          deviceId: deviceId ? { exact: deviceId } : undefined,
        },
        audio: false,
      });

      currentStreamRef.current = s;
      setStream(s);

      localStorage.setItem(LAST_DEVICE_KEY, deviceId);
    } catch (err: any) {
      console.error("openDevice error:", err);
      setStream(null);

      if (err?.name === "NotAllowedError")
        setError("Permission denied to open camera");
      else if (err?.name === "NotFoundError") setError("Device not found");
      else setError("Failed to open input device");
    } finally {
      setIsOpening(false);
      await refreshDevices();
    }
  };

  useEffect(() => {
    const tryReconnect = async () => {
      const lastId = localStorage.getItem(LAST_DEVICE_KEY);
      if (!lastId) return;

      await refreshDevices();
      const exists = devices.some((d) => d.deviceId === lastId);
      if (exists) {
        openDevice(lastId);
      }
    };

    setTimeout(tryReconnect, 300);
  }, [devices.length]);

  const setFlipH = (v: boolean) => setFlipHState(Boolean(v));
  const setFlipV = (v: boolean) => setFlipVState(Boolean(v));
  const setRotate = (v: boolean) => setRotate90State(Boolean(v));

  const value: StreamContextType = {
    stream,
    error,
    isOpening,
    devices,
    refreshDevices,
    openDevice,
    closeStream,

    flipH,
    flipV,
    rotate90,
    setFlipH,
    setFlipV,
    setRotate,
  };

  return (
    <StreamContext.Provider value={value}>{children}</StreamContext.Provider>
  );
};

export function useStream() {
  const ctx = useContext(StreamContext);
  if (!ctx) throw new Error("useStream must be used inside StreamProvider");
  return ctx;
}
