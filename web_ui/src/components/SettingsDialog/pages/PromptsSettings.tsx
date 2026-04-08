"use client";

import React from "react";
import { Button } from "@/components/ui/button";
import { ButtonGroup } from "@/components/ui/button-group";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { usePrompts } from "@/app/context/PromptProvider";

export default function PromptsSettings() {
  const {
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
    setAutoEnabled,
    setDuration,
  } = usePrompts();

  const displayIndex =
    Math.max(0, Math.min(currentIndex, Math.max(0, prompts.length - 1))) + 1;
  const maxIndex = Math.max(1, prompts.length || 1);

  return (
    <div className="flex-1 flex flex-col gap-4 max-h-[70vh]">
      <div className="flex-1 flex items-center justify-between">
        <div>
          <div className="text-sm font-medium">Current prompt</div>
          <div className="text-sm text-muted-foreground">
            Index of current prompt
          </div>
        </div>
        <div>
          <ButtonGroup>
            <Button
              onClick={prev}
              aria-label="Previous prompt"
              variant="outline"
            >
              &lt;
            </Button>

            <Input
              value={String(displayIndex)}
              onChange={(e) => setIndexOneBased(Number(e.target.value))}
              aria-label="Current prompt number"
              className="w-20 text-center"
            />

            <Button onClick={next} aria-label="Next prompt" variant="outline">
              &gt;
            </Button>
          </ButtonGroup>
        </div>
      </div>

      <div className="flex items-center justify-between">
        <div>
          <Label className="mb-0">Automatic switch</Label>
          <div className="text-sm text-muted-foreground">
            Automatically rotate prompts every N seconds
          </div>
        </div>
        <Switch
          checked={autoEnabled}
          onCheckedChange={(v) => setAutoEnabled(Boolean(v))}
        />
      </div>

      <div className="flex items-center gap-4">
        <div className="flex-1">
          <Label className="mb-0">Switch duration</Label>
          <div className="text-sm text-muted-foreground">{duration} s</div>
        </div>

        <div
          className={`w-56 ${autoEnabled ? "" : "opacity-50 pointer-events-none"}`}
        >
          <Slider
            value={[duration]}
            min={1}
            max={60}
            step={1}
            onValueChange={(v: number[] | number) => {
              const val = Array.isArray(v) ? v[0] : Number(v);
              setDuration(Math.max(1, Math.min(60, Math.round(val))));
            }}
          ></Slider>
        </div>
      </div>

      <div className="flex-1 flex flex-col min-h-0">
        <Label>Prompts</Label>
        <Textarea
          value={promptsText}
          onChange={(e) => setPromptsText(e.target.value)}
          placeholder="Enter each prompt on a new line..."
          className="flex-1 mt-2 h-full min-h-0 resize-none overflow-y-auto"
        />
      </div>

      <div className="flex-1 flex items-center justify-between text-sm text-muted-foreground">
        <div>
          {prompts.length === 0
            ? "No prompts defined."
            : `${prompts.length} prompt(s)`}
        </div>
        <div>
          Index {displayIndex} / {maxIndex}
        </div>
      </div>
    </div>
  );
}
