"use client";

import { useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";

import InputSettings from "./pages/InputSettings";
import PromptsSettings from "./pages/PromptsSettings";

const pages = [
  { id: "input", label: "Input" },
  { id: "prompts", label: "Prompts" },
];

type SettingsDialogProps = {
  open: boolean;
  onOpenChange: (open: boolean) => void;
};

export default function SettingsDialog({
  open,
  onOpenChange,
}: SettingsDialogProps) {
  const [activePage, setActivePage] = useState("input");

  const renderActive = () => {
    switch (activePage) {
      case "input":
        return <InputSettings />;
      case "prompts":
        return <PromptsSettings />;
      default:
        return null;
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="w-[90vw]! h-[90vh]! max-w-none! p-6 overflow-hidden flex flex-col max-h-[80vh]">
        <DialogHeader>
          <DialogTitle>Settings</DialogTitle>
        </DialogHeader>

        <div className="flex gap-6">
          <div className="flex flex-col gap-2 w-40 border-r pr-4">
            {pages.map((p) => (
              <Button
                key={p.id}
                variant={activePage === p.id ? "default" : "ghost"}
                className="justify-start"
                onClick={() => setActivePage(p.id)}
              >
                {p.label}
              </Button>
            ))}
          </div>

          <div className="flex-1">{renderActive()}</div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
