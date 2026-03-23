import { useWorkspaceStore } from '../../../stores/workspaceStore';
import { InlineArtifactPreview } from '../../InlineArtifactPreview';

interface FileViewerTileProps {
  filePath: string;
}

export function FileViewerTile({ filePath }: FileViewerTileProps) {
  const workspaces = useWorkspaceStore((s) => s.workspaces);
  const workspacePath = Object.keys(workspaces)[0] || '';

  if (!workspacePath) {
    return (
      <div className="flex items-center justify-center h-full text-v2-text-muted text-sm">
        No workspace available
      </div>
    );
  }

  return (
    <div className="h-full overflow-auto v2-scrollbar bg-v2-surface">
      <InlineArtifactPreview filePath={filePath} workspacePath={workspacePath} />
    </div>
  );
}
