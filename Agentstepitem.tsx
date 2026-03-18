/**
 * AgentStepItem.tsx
 *
 * Drop into: src/frontend/src/modals/IOModal/components/chatView/chatMessage/components/
 *
 * Renders a single agent step — tool calls, reasoning text, code blocks, or errors.
 * Uses Langflow's existing Tailwind/Shadcn design tokens for automatic light/dark support.
 */
import { useMemo, useState } from "react";
import {
  Wrench,
  Brain,
  CheckCircle2,
  XCircle,
  Loader2,
  ChevronDown,
  ChevronRight,
  Code2,
  AlertTriangle,
} from "lucide-react";

// ─── Type Definitions ─────────────────────────────────────────
// These mirror the ContentType union from src/frontend/src/types/chat/index.ts

interface BaseStep {
  type: string;
  duration?: number;
  header?: { title?: string; icon?: string };
}

interface ToolStep extends BaseStep {
  type: "tool_use";
  name: string;
  tool_input: Record<string, any>;
  output: any | null;
  error: any | null;
}

interface TextStep extends BaseStep {
  type: "text";
  text: string;
}

interface CodeStep extends BaseStep {
  type: "code";
  code: string;
  language: string;
  title?: string;
}

interface ErrorStep extends BaseStep {
  type: "error";
  component?: string;
  field?: string;
  reason?: string;
  solution?: string;
  traceback?: string;
}

export type AgentStep = ToolStep | TextStep | CodeStep | ErrorStep;

interface AgentStepItemProps {
  step: AgentStep;
  index: number;
  isLatest: boolean;
}

// ─── Main Export ──────────────────────────────────────────────

export default function AgentStepItem({
  step,
  index,
  isLatest,
}: AgentStepItemProps) {
  if (step.type === "tool_use") {
    return (
      <ToolStepView step={step as ToolStep} index={index} isLatest={isLatest} />
    );
  }
  if (step.type === "text") {
    return <TextStepView step={step as TextStep} index={index} />;
  }
  if (step.type === "code") {
    return <CodeStepView step={step as CodeStep} index={index} />;
  }
  if (step.type === "error") {
    return <ErrorStepView step={step as ErrorStep} index={index} />;
  }

  // Fallback for unknown content types
  return (
    <div className="flex items-center gap-2 px-3 py-1.5 text-xs text-muted-foreground">
      <Code2 className="h-3 w-3" />
      <span>
        Step {index + 1}: {step.type}
      </span>
    </div>
  );
}

// ─── Tool Step ────────────────────────────────────────────────

function ToolStepView({
  step,
  index,
  isLatest,
}: {
  step: ToolStep;
  index: number;
  isLatest: boolean;
}) {
  const [expanded, setExpanded] = useState(false);

  const status = useMemo(() => {
    if (step.error) return "error" as const;
    if (step.output !== null && step.output !== undefined) return "done" as const;
    return "running" as const;
  }, [step.output, step.error]);

  const StatusIcon = {
    running: Loader2,
    done: CheckCircle2,
    error: XCircle,
  }[status];

  const statusColor = {
    running: "text-blue-500",
    done: "text-green-500",
    error: "text-red-500",
  }[status];

  const bgClass = {
    running:
      "border-blue-200 bg-blue-50/50 dark:border-blue-800 dark:bg-blue-950/30",
    done: "border-border/50 bg-background/50",
    error:
      "border-red-200 bg-red-50/50 dark:border-red-800 dark:bg-red-950/30",
  }[status];

  const toolDisplayName =
    step.name?.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()) ??
    "Unknown Tool";

  return (
    <div className={`group rounded-lg border transition-all duration-200 ${bgClass}`}>
      {/* Header */}
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-center gap-2 px-3 py-2 text-left text-sm"
      >
        <StatusIcon
          className={`h-4 w-4 flex-shrink-0 ${statusColor} ${
            status === "running" ? "animate-spin" : ""
          }`}
        />
        <Wrench className="h-3.5 w-3.5 flex-shrink-0 text-muted-foreground" />
        <span className="font-medium truncate flex-1">{toolDisplayName}</span>
        {step.duration != null && (
          <span className="text-xs text-muted-foreground tabular-nums">
            {step.duration < 1000
              ? `${step.duration}ms`
              : `${(step.duration / 1000).toFixed(1)}s`}
          </span>
        )}
        {expanded ? (
          <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
        )}
      </button>

      {/* Expanded details */}
      {expanded && (
        <div className="border-t border-border/50 px-3 py-2 space-y-2">
          {step.tool_input && Object.keys(step.tool_input).length > 0 && (
            <div>
              <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-1">
                Input
              </div>
              <pre className="text-xs bg-muted/50 rounded p-2 overflow-x-auto max-h-40 overflow-y-auto font-mono">
                {JSON.stringify(step.tool_input, null, 2)}
              </pre>
            </div>
          )}
          {step.output != null && (
            <div>
              <div className="text-[10px] font-semibold uppercase tracking-wider text-muted-foreground mb-1">
                Output
              </div>
              <pre className="text-xs bg-muted/50 rounded p-2 overflow-x-auto max-h-40 overflow-y-auto font-mono">
                {typeof step.output === "string"
                  ? step.output
                  : JSON.stringify(step.output, null, 2)}
              </pre>
            </div>
          )}
          {step.error != null && (
            <div>
              <div className="text-[10px] font-semibold uppercase tracking-wider text-red-500 mb-1">
                Error
              </div>
              <pre className="text-xs bg-red-50 dark:bg-red-950/30 text-red-700 dark:text-red-300 rounded p-2 overflow-x-auto font-mono">
                {typeof step.error === "string"
                  ? step.error
                  : JSON.stringify(step.error, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Text / Reasoning Step ────────────────────────────────────

function TextStepView({ step, index }: { step: TextStep; index: number }) {
  const [expanded, setExpanded] = useState(true);
  const isThinking =
    step.header?.title?.toLowerCase().includes("think") ||
    step.text.length > 200;

  return (
    <div className="rounded-lg border border-border/30 bg-background/30">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm"
      >
        <Brain className="h-3.5 w-3.5 flex-shrink-0 text-purple-500" />
        <span className="text-muted-foreground text-xs flex-1 truncate">
          {step.header?.title || (isThinking ? "Reasoning" : "Response")}
        </span>
        {step.duration != null && (
          <span className="text-xs text-muted-foreground tabular-nums">
            {step.duration < 1000
              ? `${step.duration}ms`
              : `${(step.duration / 1000).toFixed(1)}s`}
          </span>
        )}
        {expanded ? (
          <ChevronDown className="h-3 w-3 text-muted-foreground" />
        ) : (
          <ChevronRight className="h-3 w-3 text-muted-foreground" />
        )}
      </button>
      {expanded && (
        <div className="border-t border-border/30 px-3 py-2">
          <p className="text-xs text-foreground/80 whitespace-pre-wrap leading-relaxed max-h-60 overflow-y-auto">
            {step.text}
          </p>
        </div>
      )}
    </div>
  );
}

// ─── Code Step ────────────────────────────────────────────────

function CodeStepView({ step, index }: { step: CodeStep; index: number }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="rounded-lg border border-border/30 bg-background/30">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm"
      >
        <Code2 className="h-3.5 w-3.5 flex-shrink-0 text-amber-500" />
        <span className="text-muted-foreground text-xs flex-1 truncate">
          {step.title || `Code (${step.language})`}
        </span>
        {expanded ? (
          <ChevronDown className="h-3 w-3 text-muted-foreground" />
        ) : (
          <ChevronRight className="h-3 w-3 text-muted-foreground" />
        )}
      </button>
      {expanded && (
        <div className="border-t border-border/30">
          <pre className="text-xs p-3 overflow-x-auto max-h-60 overflow-y-auto bg-muted/30 font-mono">
            <code>{step.code}</code>
          </pre>
        </div>
      )}
    </div>
  );
}

// ─── Error Step ───────────────────────────────────────────────

function ErrorStepView({ step, index }: { step: ErrorStep; index: number }) {
  const [expanded, setExpanded] = useState(true);

  return (
    <div className="rounded-lg border border-red-200 dark:border-red-800 bg-red-50/50 dark:bg-red-950/30">
      <button
        onClick={() => setExpanded(!expanded)}
        className="flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm"
      >
        <AlertTriangle className="h-3.5 w-3.5 flex-shrink-0 text-red-500" />
        <span className="text-red-700 dark:text-red-300 text-xs flex-1 truncate">
          Error{step.component ? ` in ${step.component}` : ""}
        </span>
      </button>
      {expanded && (
        <div className="border-t border-red-200/50 dark:border-red-800/50 px-3 py-2 space-y-1">
          {step.reason && (
            <p className="text-xs text-red-700 dark:text-red-300">
              {step.reason}
            </p>
          )}
          {step.solution && (
            <p className="text-xs text-muted-foreground">💡 {step.solution}</p>
          )}
        </div>
      )}
    </div>
  );
}
