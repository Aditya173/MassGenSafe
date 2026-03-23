import { useState } from 'react';
import { cn } from '../../../../lib/utils';
import type { ContentMessage } from '../../../../stores/v2/messageStore';

interface ContentMessageViewProps {
  message: ContentMessage;
}

export function ContentMessageView({ message }: ContentMessageViewProps) {
  const isThinking = message.contentType === 'thinking';
  const [expanded, setExpanded] = useState(!isThinking);

  const content = message.content.trim();
  if (!content) return null;

  const lines = content.split('\n');
  const hasMore = lines.length > 3 || content.length > 300;
  const preview = lines.slice(0, 2).join('\n');
  const previewTruncated = preview.length > 200 ? preview.slice(0, 200) + '\u2026' : preview;

  // Thinking/reasoning — collapsible, purple left-border
  if (isThinking) {
    return (
      <div className="px-4 py-1">
        <div
          className={cn(
            'border-l-2 border-violet-400/30 pl-3 cursor-pointer',
            'hover:border-violet-400/50 transition-colors duration-150',
          )}
          onClick={() => setExpanded(!expanded)}
        >
          <div className="flex items-center gap-1.5 mb-0.5">
            <svg
              className={cn(
                'w-2.5 h-2.5 text-v2-text-muted transition-transform duration-150 shrink-0',
                expanded && 'rotate-90'
              )}
              viewBox="0 0 12 12"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
            >
              <path d="M4 2l4 4-4 4" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
            <span className="text-[11px] font-medium uppercase tracking-wider text-violet-400/70">
              Reasoning
            </span>
          </div>
          {expanded ? (
            <pre className="whitespace-pre-wrap text-[13px] leading-relaxed break-words text-v2-text-muted italic animate-v2-fade-in">
              {content}
            </pre>
          ) : (
            <p className="text-xs text-v2-text-muted italic truncate">
              {previewTruncated}
            </p>
          )}
        </div>
      </div>
    );
  }

  // Regular content — accent left-border, flush left
  if (!hasMore) {
    return (
      <div className="px-4 py-1">
        <div className="border-l-2 border-v2-text-muted/20 pl-3">
          <p className="text-sm text-v2-text leading-relaxed">
            {content}
          </p>
        </div>
      </div>
    );
  }

  // Longer content: collapsible with accent border
  return (
    <div className="px-4 py-1">
      <div
        className="border-l-2 border-v2-text-muted/20 pl-3 cursor-pointer"
        onClick={() => setExpanded(!expanded)}
      >
        {expanded ? (
          <div className="animate-v2-fade-in">
            <pre className="whitespace-pre-wrap text-sm text-v2-text leading-relaxed break-words">
              {content}
            </pre>
            <button className="text-xs text-v2-text-muted mt-1 hover:text-v2-text transition-colors">
              Show less
            </button>
          </div>
        ) : (
          <p className="text-sm text-v2-text leading-relaxed">
            {previewTruncated}
            {hasMore && (
              <span className="text-v2-text-muted ml-1">
                (+{lines.length - 2} lines)
              </span>
            )}
          </p>
        )}
      </div>
    </div>
  );
}
