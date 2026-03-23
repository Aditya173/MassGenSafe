import type { RoundDividerMessage } from '../../../../stores/v2/messageStore';

interface RoundDividerViewProps {
  message: RoundDividerMessage;
}

export function RoundDividerView({ message }: RoundDividerViewProps) {
  return (
    <div className="flex items-center gap-3 px-4 py-3">
      <div className="flex-1 h-px bg-v2-border" />
      <span className="text-[11px] font-medium uppercase tracking-wider text-v2-text-muted shrink-0">
        {message.label}
      </span>
      <div className="flex-1 h-px bg-v2-border" />
    </div>
  );
}
