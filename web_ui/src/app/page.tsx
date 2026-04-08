"use client";

import { StreamProvider } from "./context/StreamProvider";
import { ModelProvider } from "./context/ModelProvider";
import { PromptProvider } from "./context/PromptProvider";
import MainStreamViewer from "@/components/MainStreamViewer";

export default function Page() {
  return (
    <StreamProvider>
      <ModelProvider>
        <PromptProvider>
          <main className="flex h-screen w-full items-center justify-center p-4">
            <MainStreamViewer />
          </main>
        </PromptProvider>
      </ModelProvider>
    </StreamProvider>
  );
}
