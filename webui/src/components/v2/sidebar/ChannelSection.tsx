import { useState, useEffect, useRef } from 'react';
import { cn } from '../../../lib/utils';
import { useAgentStore } from '../../../stores/agentStore';
import { useTileStore } from '../../../stores/v2/tileStore';
import { getAgentColor } from '../../../utils/agentColors';
import { SidebarItem } from './SessionSection';

interface ChannelSectionProps {
  collapsed: boolean;
}

export function ChannelSection({ collapsed }: ChannelSectionProps) {
  const agents = useAgentStore((s) => s.agents);
  const agentOrder = useAgentStore((s) => s.agentOrder);
  const { tiles, setTile } = useTileStore();

  const activeAgentTileId = tiles.find((t) => t.type === 'agent-channel')?.targetId;

  // Stagger animation: only on first agent arrival
  const [hasAnimated, setHasAnimated] = useState(false);
  const prevCountRef = useRef(0);

  useEffect(() => {
    if (agentOrder.length > 0 && prevCountRef.current === 0) {
      const timeout = setTimeout(() => setHasAnimated(true), agentOrder.length * 80 + 300);
      return () => clearTimeout(timeout);
    }
    prevCountRef.current = agentOrder.length;
  }, [agentOrder.length]);

  const handleChannelClick = (agentId: string) => {
    const agent = agents[agentId];
    setTile({
      id: `channel-${agentId}`,
      type: 'agent-channel',
      targetId: agentId,
      label: agent?.modelName || agentId,
    });
  };

  return (
    <div className="py-1">
      {!collapsed && (
        <div className="flex items-center px-2 py-1">
          <span className="text-[11px] font-semibold uppercase tracking-wide text-v2-text-muted">
            Channels
          </span>
        </div>
      )}

      <div className="space-y-0.5">
        {agentOrder.map((agentId, index) => {
          const agent = agents[agentId];
          if (!agent) return null;
          const isActive = activeAgentTileId === agentId;
          const agentColor = getAgentColor(agentId, agentOrder);
          const isWorking = agent.status === 'working' || agent.status === 'voting';
          const shouldAnimate = !hasAnimated;

          return (
            <div
              key={agentId}
              className={shouldAnimate ? 'opacity-0 animate-v2-stagger-fade-in' : undefined}
              style={shouldAnimate ? { animationDelay: `${index * 80}ms`, animationFillMode: 'forwards' } : undefined}
            >
              <SidebarItem
                collapsed={collapsed}
                active={isActive}
                onClick={() => handleChannelClick(agentId)}
                icon={
                  <span
                    className={cn('w-2 h-2 rounded-full', isWorking && 'animate-pulse')}
                    style={{ backgroundColor: agentColor.hex }}
                  />
                }
                label={formatChannelLabel(agentId, agent.modelName)}
              />
            </div>
          );
        })}

        {agentOrder.length === 0 && !collapsed && (
          <p className="text-xs text-v2-text-muted px-2 py-2 italic">
            No agents yet
          </p>
        )}
      </div>
    </div>
  );
}

function formatChannelLabel(agentId: string, modelName?: string): string {
  const name = agentId.replace(/_/g, ' ');
  if (modelName) {
    return `${name} (${modelName})`;
  }
  return name;
}
