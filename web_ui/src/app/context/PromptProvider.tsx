"use client";

import React, {
  createContext,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { useModel } from "./ModelProvider";

type PromptState = {
  promptsText: string;
  prompts: string[];
  currentIndex: number;
  autoEnabled: boolean;
  duration: number;
  sending: boolean;

  setPromptsText: (text: string) => void;
  prev: () => void;
  next: () => void;
  setIndexOneBased: (oneBased: number) => void;
  setAutoEnabled: (v: boolean) => void;
  setDuration: (s: number) => void;
};

const PromptContext = createContext<PromptState | undefined>(undefined);

export const PromptProvider: React.FC<{ children: React.ReactNode }> = ({
  children,
}) => {
  const { selectedModelId, models } = useModel();
  const selectedModel = models.find((m) => m.id === selectedModelId);

  const [promptsText, setPromptsTextState] = useState<string>("");
  const [prompts, setPrompts] = useState<string[]>([]);
  const [currentIndex, setCurrentIndex] = useState<number>(0);
  const [autoEnabled, setAutoEnabledState] = useState<boolean>(false);
  const [duration, setDurationState] = useState<number>(5);
  const [sending, setSending] = useState<boolean>(false);

  const storageKey = useMemo(
    () => `prompts:${selectedModelId ?? "global"}`,
    [selectedModelId],
  );
  useEffect(() => {
    try {
      const raw = localStorage.getItem(storageKey);
      if (raw) {
        const parsed = JSON.parse(raw);
        if (Array.isArray(parsed.prompts)) {
          setPrompts(parsed.prompts);
          setPromptsTextState(parsed.prompts.join("\n"));
        } else {
          setPrompts([]);
          setPromptsTextState("");
        }
        setCurrentIndex(
          typeof parsed.currentIndex === "number" ? parsed.currentIndex : 0,
        );
        setAutoEnabledState(Boolean(parsed.autoEnabled));
        setDurationState(
          typeof parsed.duration === "number" ? parsed.duration : 5,
        );
        return;
      }
    } catch (e) {
      console.warn("PromptProvider load error:", e);
    }
    setPrompts([]);
    setPromptsTextState("");
    setCurrentIndex(0);
    setAutoEnabledState(false);
    setDurationState(5);
  }, [storageKey]);

  useEffect(() => {
    try {
      const payload = {
        prompts,
        currentIndex,
        autoEnabled,
        duration,
      };
      localStorage.setItem(storageKey, JSON.stringify(payload));
    } catch (e) {
      console.warn("PromptProvider save error:", e);
    }
  }, [storageKey, prompts, currentIndex, autoEnabled, duration]);

  const parseLines = (text: string) => text.replace(/\r/g, "").split("\n");

  const setPromptsText = (text: string) => {
    setPromptsTextState(text);
    const lines = parseLines(text);
    setPrompts(lines);

    setCurrentIndex((idx) => {
      const max = Math.max(0, lines.length - 1);
      const newIdx = Math.min(idx, max);

      void sendPromptToServerRef.current(lines[newIdx] ?? "");

      return newIdx;
    });
  };

  const sendPromptToServerRef = useRef(async (pt: string) => {});
  sendPromptToServerRef.current = async (pt: string) => {
    if (!selectedModel) return;
    try {
      setSending(true);
      const url = `${selectedModel.baseHttp}/prompt`;
      await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: pt }),
      });
    } catch (err) {
      console.error("sendPrompt error", err);
    } finally {
      setSending(false);
    }
  };

  useEffect(() => {
    if (!prompts || prompts.length === 0) return;
    const idx = Math.max(0, Math.min(currentIndex, prompts.length - 1));
    const text = prompts[idx] ?? "";
    void sendPromptToServerRef.current(text);
  }, [currentIndex, selectedModelId]);

  const prev = () => {
    setCurrentIndex((idx) => {
      if (prompts.length === 0) return 0;
      return idx <= 0 ? prompts.length - 1 : idx - 1;
    });
  };
  const next = () => {
    setCurrentIndex((idx) =>
      prompts.length === 0 ? 0 : (idx + 1) % prompts.length,
    );
  };
  const setIndexOneBased = (oneBased: number) => {
    const parsed = Number(oneBased);
    if (Number.isNaN(parsed)) return;
    const target =
      Math.max(1, Math.min(parsed, Math.max(1, prompts.length || 1))) - 1;
    setCurrentIndex(target);
  };

  const intervalRef = useRef<number | null>(null);
  useEffect(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    if (autoEnabled && prompts.length > 0) {
      const ms = Math.max(1000, Math.min(duration * 1000, 60_000));
      intervalRef.current = window.setInterval(() => {
        setCurrentIndex((idx) =>
          prompts.length === 0 ? 0 : (idx + 1) % prompts.length,
        );
      }, ms);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
    };
  }, [autoEnabled, duration, prompts.length]);

  const value: PromptState = {
    promptsText,
    prompts,
    currentIndex,
    autoEnabled,
    duration,
    sending,
    setPromptsText,
    prev,
    next,
    setIndexOneBased,
    setAutoEnabled: setAutoEnabledState,
    setDuration: setDurationState,
  };

  return (
    <PromptContext.Provider value={value}>{children}</PromptContext.Provider>
  );
};

export function usePrompts() {
  const ctx = useContext(PromptContext);
  if (!ctx) throw new Error("usePrompts must be used inside PromptProvider");
  return ctx;
}
