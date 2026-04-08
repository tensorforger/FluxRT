"use client";

import { useEffect, useRef, useState } from "react";
import { Settings } from "lucide-react";
import { Button } from "@/components/ui/button";
import SettingsDialog from "./SettingsDialog/SettingsDialog";
import { useStream } from "@/app/context/StreamProvider";
import { useModel } from "@/app/context/ModelProvider";

export default function MainStreamViewer({
  outputFrameUrl,
}: { outputFrameUrl?: string | null } = {}) {
  const [openSettings, setOpenSettings] = useState(false);
  const { stream } = useStream();
  const { latestFrameUrl } = useModel();

  const videoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    if (stream) {
      v.srcObject = stream;
      v.play().catch(() => {});
    } else {
      if (v.srcObject) {
        v.srcObject = null;
      }
    }
  }, [stream]);

  const showOverlay = false;

  return (
    <div className="relative w-full h-full bg-background flex items-center justify-center">
      <div className="relative w-full h-full max-w-6xl">
        <Button
          variant="ghost"
          size="icon"
          className="absolute top-4 right-4 z-20 text-white/50"
          onClick={() => setOpenSettings(true)}
        >
          <Settings className="h-5 w-5" />
        </Button>

        <div className="w-full h-full bg-muted rounded-2xl overflow-hidden flex items-center justify-center border">
          {latestFrameUrl || outputFrameUrl ? (
            <img
              src={latestFrameUrl ?? outputFrameUrl ?? undefined}
              alt="Model output"
              className="object-cover w-full h-full rounded-2xl"
            />
          ) : (
            <div className="w-full h-full flex items-center justify-center">
              <video
                ref={videoRef}
                className={`object-contain max-w-full max-h-full ${stream ? "block" : "hidden"} rounded-2xl`}
                playsInline
                muted
                autoPlay
              />
              {!stream && (
                <div className="text-muted-foreground">No output provided</div>
              )}
            </div>
          )}

          {/* Sometimes you need to overlay something (logo or text). Here is an example. */}
          {showOverlay && (
            <div className="absolute inset-0 pointer-events-none flex flex-col justify-between p-4 text-2xl">
              <div className="absolute bottom-4 left-4 bg-muted/30 backdrop-blur-md px-2 py-2 rounded-4xl shadow-lg border border-border glow">
                <span className="font-bold text-white mr-2">Some text</span>
              </div>

              <div className="absolute bottom-4 right-4 bg-muted/30 backdrop-blur-md px-2 py-2 rounded-xl shadow-lg border border-border glow">
                <img src="/image.png" alt="QR Code" className="w-32 h-32" />
              </div>
            </div>
          )}
        </div>
      </div>

      <SettingsDialog open={openSettings} onOpenChange={setOpenSettings} />
    </div>
  );
}
