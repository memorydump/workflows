/**
 * AgentScratchpad.tsx
 *
 * Drop into: src/frontend/src/modals/IOModal/components/chatView/chatMessage/components/
 *
 * Container component that reads content_blocks from a ChatMessageType
 * and renders each step using AgentStepItem.
 *
 * Usage in newChatMessage.tsx:
 *
 *   import AgentScratchpad from "./components/AgentScratchpad";
 *
 *   // Replace loading dots with:
 *   <AgentScratchpad
 *     contentBlocks={message.content_blocks}
 *     isStreaming={!!message.stream_url || !message.text}
 *   />
 */
import { useEffect, useRef, useMemo } from "react";
import { Bot, Loader2 } from "lucide-react";
import AgentStepItem from "./AgentStepItem";
import type { AgentStep } from "./AgentStepItem";

// ─── Types ────────────────────────────────────────────────────
// Matches ContentBlock from src/frontend/src/types/chat/index.ts

interface ContentBlock {
  title: string;
  contents: Array<{
    type: string;
    [key: string]: any;
  }>;
}

interface AgentScratchpadProps {
  /** The content_blocks array from ChatMessageType */
  contentBlocks: ContentBlock[] | undefined | null;
  /** Whether the message is still being streamed */
  isStreaming: boolean;
}

// ─── Component ────────────────────────────────────────────────

export default function AgentScratchpad({
  contentBlocks,
  isStreaming,
}: AgentScratchpadProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  // Flatten all content blocks into a single ordered list of steps
  const agentSteps = useMemo(() => {
    if (!contentBlocks || contentBlocks.length === 0) return [];

    return contentBlocks.flatMap((block) =>
      (block.contents || []).map((content, idx) => ({
        ...content,
        _blockTitle: block.title,
        _index: idx,
      })),
    );
  }, [contentBlocks]);

  // Auto-scroll to latest step
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [agentSteps.length]);

  // ─── State: No steps yet, still streaming → "thinking" indicator ──
  if (agentSteps.length === 0 && isStreaming) {
    return (
      <div className="flex items-center gap-2.5 px-1 py-2">
        <div className="relative flex items-center justify-center">
          <Loader2 className="h-4 w-4 animate-spin text-primary/60" />
        </div>
        <span className="text-sm text-muted-foreground animate-pulse">
          Agent is thinking…
        </span>
      </div>
    );
  }

  // ─── State: No steps, not streaming → nothing to show ─────────────
  if (agentSteps.length === 0) {
    return null;
  }

  // ─── State: Steps available → render the scratchpad ───────────────
  return (
    <div className="w-full">
      {/* Section Header */}
      <div className="flex items-center gap-2 mb-2">
        <Bot className="h-3.5 w-3.5 text-primary/70" />
        <span className="text-[11px] font-semibold uppercase tracking-wider text-muted-foreground">
          Agent Steps
        </span>
        <span className="text-[10px] text-muted-foreground/60 tabular-nums">
          ({agentSteps.length})
        </span>
        {isStreaming && (
          <Loader2 className="h-3 w-3 animate-spin text-primary/50 ml-auto" />
        )}
      </div>

      {/* Steps List */}
      <div
        ref={scrollRef}
        className="space-y-1.5 max-h-[400px] overflow-y-auto pr-1"
        style={{
          scrollbarWidth: "thin",
          scrollbarColor: "hsl(var(--muted-foreground) / 0.2) transparent",
        }}
      >
        {agentSteps.map((step, idx) => (
          <AgentStepItem
            key={`${step.type}-${step._blockTitle}-${idx}`}
            step={step as AgentStep}
            index={idx}
            isLatest={idx === agentSteps.length - 1}
          />
        ))}
      </div>
    </div>
  );
}
