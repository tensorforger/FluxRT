"use client";

import React, { useEffect, useRef, useState } from "react";
import { useStream } from "@/app/context/StreamProvider";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Label } from "@/components/ui/label";
import { Button } from "@/components/ui/button";
import { Switch } from "@/components/ui/switch";

export default function InputSettings() {
  const {
    devices,
    refreshDevices,
    openDevice,
    stream,
    error,
    isOpening,
    flipH,
    flipV,
    rotate90,
    setFlipH,
    setFlipV,
    setRotate,
  } = useStream();
  const [selectedDeviceId, setSelectedDeviceId] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);

  useEffect(() => {
    refreshDevices();
  }, []);

  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    if (stream) {
      v.srcObject = stream;
      v.play().catch(() => {});
    } else {
      if (v.srcObject) v.srcObject = null;
    }
  }, [stream]);

  const handleSelect = async (deviceId: string) => {
    setSelectedDeviceId(deviceId);
    await openDevice(deviceId);
  };

  const transformStr = `scaleX(${flipH ? -1 : 1}) scaleY(${flipV ? -1 : 1})${rotate90 ? " rotate(90deg)" : ""}`;
  const transformStyle = {
    transform: transformStr,
    transformOrigin: "center center",
  } as React.CSSProperties;

  return (
    <div className="flex flex-col gap-4">
      <div>
        <Label>Camera device</Label>
        <div className="mt-2">
          <Select
            value={selectedDeviceId ?? ""}
            onValueChange={(v) => handleSelect(v)}
          >
            <SelectTrigger className="w-[260px]">
              <SelectValue
                placeholder={
                  devices.length ? "Select camera..." : "No cameras found"
                }
              />
            </SelectTrigger>
            <SelectContent>
              {devices.length === 0 ? (
                <SelectItem value="__no_devices__" disabled>
                  No devices
                </SelectItem>
              ) : (
                devices.map((d, i) => {
                  const hasId = d.deviceId && d.deviceId.trim() !== "";
                  return hasId ? (
                    <SelectItem key={d.deviceId} value={d.deviceId}>
                      {d.label || `Camera (${d.deviceId.slice(0, 6)})`}
                    </SelectItem>
                  ) : (
                    <SelectItem
                      key={`unknown-${i}`}
                      value={`__unknown_${i}`}
                      disabled
                    >
                      {d.label || "No permission to use camera"}
                    </SelectItem>
                  );
                })
              )}
            </SelectContent>
          </Select>
        </div>
      </div>

      <div>
        <Label>Flip</Label>
        <div className="mt-2 items-center gap-2 p-1">
          <div className="flex items-center gap-2 pb-1">
            <Switch
              checked={flipH}
              onCheckedChange={(v) => setFlipH(Boolean(v))}
            />
            <div className="text-sm">Horizontally</div>
          </div>
          <div className="flex items-center gap-2 pb-1">
            <Switch
              checked={flipV}
              onCheckedChange={(v) => setFlipV(Boolean(v))}
            />
            <div className="text-sm">Vertically</div>
          </div>
        </div>
      </div>

      <div>
        <Label>Rotate</Label>
        <div className="mt-2 items-center gap-2 p-1">
          <div className="flex items-center gap-2">
            <Switch
              checked={rotate90}
              onCheckedChange={(v) => setRotate(Boolean(v))}
            />
            <div className="text-sm">Clockwise</div>
          </div>
        </div>
      </div>

      <div>
        <Label>Preview</Label>
        <div className="mt-2 w-64 h-40 bg-black rounded-md overflow-hidden flex items-center justify-center">
          <video
            ref={videoRef}
            className={`w-full h-full object-cover ${stream ? "block" : "hidden"}`}
            playsInline
            muted
            style={transformStyle}
          />
          {!stream && !error && (
            <div className="text-sm text-muted-foreground">
              Device is not selected
            </div>
          )}
          {error && <div className="text-sm text-red-500">{error}</div>}
        </div>
      </div>

      <div className="text-sm text-muted-foreground">
        {isOpening && <span>Opening device...</span>}
      </div>
    </div>
  );
}
