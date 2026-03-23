import { useState } from 'react';
import { cn } from '../../../lib/utils';
import { useAgentStore } from '../../../stores/agentStore';

const MAX_DISPLAY_LENGTH = 80;

function truncateQuestion(q: string): string {
  const single = q.replace(/\n/g, ' ').trim();
  if (single.length <= MAX_DISPLAY_LENGTH) return single;
  return single.slice(0, MAX_DISPLAY_LENGTH - 1) + '\u2026';
}

export function PromptBanner() {
  const question = useAgentStore((s) => s.question);
  const turnNumber = useAgentStore((s) => s.turnNumber);
  const [expanded, setExpanded] = useState(false);

  if (!question) return null;

  return (
    <div className="relative shrink-0">
      {/* Thin inline bar — does not overlap content */}
      <div
        data-testid="prompt-banner"
        className={cn(
          'flex items-center gap-2 px-4 py-1 cursor-pointer',
          'bg-v2-surface border-b border-v2-border-subtle',
          'text-[11px] text-v2-text-muted',
          'hover:bg-v2-surface-raised transition-colors duration-100'
        )}
        title="Click to view full prompt"
        onClick={() => setExpanded(!expanded)}
      >
        <span className="text-v2-accent font-semibold shrink-0">
          Turn {turnNumber}
        </span>
        <span className="text-v2-border-subtle">|</span>
        <span className="italic truncate">
          {truncateQuestion(question)}
        </span>
      </div>

      {/* Expanded overlay — drops down from the bar */}
      {expanded && (
        <div
          data-testid="prompt-expanded"
          className={cn(
            'absolute top-full left-0 right-0 z-20',
            'border-b border-v2-border shadow-lg',
            'bg-v2-surface-raised/95 backdrop-blur-sm',
            'animate-v2-fade-in'
          )}
        >
          <div className="flex items-start justify-between px-4 py-2.5">
            <p className="text-sm text-v2-text whitespace-pre-wrap break-words flex-1 max-h-48 overflow-y-auto v2-scrollbar">
              {question}
            </p>
            <button
              data-testid="prompt-expanded-close"
              onClick={(e) => { e.stopPropagation(); setExpanded(false); }}
              className={cn(
                'flex items-center justify-center w-5 h-5 rounded shrink-0 ml-2',
                'text-v2-text-muted hover:text-v2-text hover:bg-v2-sidebar-hover',
                'transition-colors duration-150'
              )}
            >
              <svg width="10" height="10" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M2 2l8 8M10 2l-8 8" strokeLinecap="round" />
              </svg>
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
