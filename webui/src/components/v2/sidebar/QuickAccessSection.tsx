import { useTileStore, type TileType } from '../../../stores/v2/tileStore';
import { SidebarItem } from './SessionSection';

/** The three workspace tile types — only one can be open at a time */
const WORKSPACE_TILE_TYPES: TileType[] = ['workspace-browser', 'timeline-view', 'vote-results'];

interface QuickAccessSectionProps {
  collapsed: boolean;
}

export function QuickAccessSection({ collapsed }: QuickAccessSectionProps) {
  const tiles = useTileStore((s) => s.tiles);
  const addTile = useTileStore((s) => s.addTile);
  const removeTile = useTileStore((s) => s.removeTile);

  // Which workspace tile is currently open (if any)
  const openWorkspaceTile = tiles.find((t) => WORKSPACE_TILE_TYPES.includes(t.type));

  const toggleTile = (id: string, type: TileType, label: string, targetId: string) => {
    if (openWorkspaceTile?.type === type) {
      // Same tile — close it
      removeTile(openWorkspaceTile.id);
    } else {
      // Different tile or none open — remove existing workspace tile first, then add new
      if (openWorkspaceTile) {
        removeTile(openWorkspaceTile.id);
      }
      addTile({ id, type, targetId, label });
    }
  };

  return (
    <div className="py-1">
      {!collapsed && (
        <div className="flex items-center px-2 py-1">
          <span className="text-[11px] font-semibold uppercase tracking-wide text-v2-text-muted">
            Workspace
          </span>
        </div>
      )}

      <div className="space-y-0.5">
        <SidebarItem
          collapsed={collapsed}
          icon={<FolderIcon />}
          label="Browse files"
          active={openWorkspaceTile?.type === 'workspace-browser'}
          onClick={() => toggleTile('workspace-browser', 'workspace-browser', 'Browse files', 'workspace')}
        />
        <SidebarItem
          collapsed={collapsed}
          icon={<TimelineIcon />}
          label="Timeline"
          active={openWorkspaceTile?.type === 'timeline-view'}
          onClick={() => toggleTile('timeline-view', 'timeline-view', 'Timeline', 'timeline')}
        />
        <SidebarItem
          collapsed={collapsed}
          icon={<VoteIcon />}
          label="Vote results"
          active={openWorkspaceTile?.type === 'vote-results'}
          onClick={() => toggleTile('vote-results', 'vote-results', 'Vote results', 'votes')}
        />
      </div>
    </div>
  );
}

function FolderIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M2 4.5A1.5 1.5 0 013.5 3h3l1.5 1.5h4.5A1.5 1.5 0 0114 6v5.5a1.5 1.5 0 01-1.5 1.5h-9A1.5 1.5 0 012 11.5V4.5z" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  );
}

function TimelineIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M2 3v10M6 5v6M10 4v8M14 6v4" strokeLinecap="round" />
    </svg>
  );
}

function VoteIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M3 12V8M6.5 12V5M10 12V3M13.5 12V7" strokeLinecap="round" />
    </svg>
  );
}
