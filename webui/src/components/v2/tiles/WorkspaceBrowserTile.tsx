import { useState, useEffect, useMemo, useRef } from 'react';
import { Panel, Group, Separator } from 'react-resizable-panels';
import { ExternalLink, Eye, X } from 'lucide-react';
import { cn } from '../../../lib/utils';
import { useWorkspaceStore, type WorkspaceFileInfo } from '../../../stores/workspaceStore';
import { useAgentStore } from '../../../stores/agentStore';
import { useTileStore } from '../../../stores/v2/tileStore';
import { getAgentColor } from '../../../utils/agentColors';
import { canPreviewFile } from '../../../utils/artifactTypes';
import { InlineArtifactPreview } from '../../InlineArtifactPreview';

export function WorkspaceBrowserTile() {
  const workspaces = useWorkspaceStore((s) => s.workspaces);
  const agentOrder = useAgentStore((s) => s.agentOrder);
  const addTile = useTileStore((s) => s.addTile);

  const workspacePaths = Object.keys(workspaces);
  const [selectedWorkspace, setSelectedWorkspace] = useState<string>('');
  const [selectedFile, setSelectedFile] = useState<WorkspaceFileInfo | null>(null);
  const lastAutoPreviewKeyRef = useRef<string | null>(null);

  const effectiveWorkspace =
    workspacePaths.includes(selectedWorkspace) ? selectedWorkspace : workspacePaths[0] || '';

  const files = workspaces[effectiveWorkspace]?.files || [];

  // Clear selected file when workspace changes
  useEffect(() => {
    setSelectedFile(null);
  }, [effectiveWorkspace]);

  useEffect(() => {
    if (selectedFile && !files.some((file) => file.path === selectedFile.path)) {
      setSelectedFile(null);
    }
  }, [files, selectedFile]);

  useEffect(() => {
    if (selectedFile) return;

    const mainPreviewableFile = findMainPreviewableFile(files);
    if (!mainPreviewableFile) return;

    const autoPreviewKey = `${effectiveWorkspace}:${mainPreviewableFile.path}`;
    if (lastAutoPreviewKeyRef.current === autoPreviewKey) return;

    lastAutoPreviewKeyRef.current = autoPreviewKey;
    setSelectedFile(mainPreviewableFile);
  }, [effectiveWorkspace, files, selectedFile]);

  const handleFileClick = (file: WorkspaceFileInfo) => {
    setSelectedFile(file);
  };

  const handleOpenInTile = () => {
    if (!selectedFile) return;
    addTile({
      id: `file-${selectedFile.path}`,
      type: 'file-viewer',
      targetId: selectedFile.path,
      label: selectedFile.path.split('/').pop() || selectedFile.path,
    });
  };

  const wsStatus = useWorkspaceStore((s) => s.wsStatus);
  const sessionId = useAgentStore((s) => s.sessionId);

  if (workspacePaths.length === 0) {
    const isWaiting = !!sessionId && (wsStatus === 'connecting' || wsStatus === 'connected');
    return (
      <div className="flex flex-col items-center justify-center h-full text-v2-text-muted text-sm gap-2">
        {isWaiting ? (
          <>
            <svg className="w-5 h-5 animate-spin" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="8" cy="8" r="6" strokeDasharray="20" strokeDashoffset="5" />
            </svg>
            <span>Waiting for workspace data...</span>
          </>
        ) : (
          <span>No workspace files available</span>
        )}
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full bg-v2-base">
      {/* Workspace selector (multi-agent) */}
      {workspacePaths.length > 1 && (
        <div className="flex items-center gap-1 px-3 py-2 border-b border-v2-border shrink-0">
          {workspacePaths.map((path, index) => {
            const agentId = agentOrder[index];
            const color = agentId
              ? getAgentColor(agentId, agentOrder)
              : undefined;
            const isSelected = path === effectiveWorkspace;
            return (
              <button
                key={path}
                onClick={() => setSelectedWorkspace(path)}
                className={cn(
                  'px-2 py-1 text-xs rounded transition-colors',
                  isSelected
                    ? 'text-v2-text font-medium'
                    : 'text-v2-text-muted hover:text-v2-text hover:bg-v2-sidebar-hover'
                )}
                style={
                  isSelected && color
                    ? { backgroundColor: `${color.hex}20` }
                    : undefined
                }
              >
                <span
                  className="inline-block w-2 h-2 rounded-full mr-1.5"
                  style={{ backgroundColor: color?.hex || '#80848E' }}
                />
                {agentId || `Workspace ${index + 1}`}
              </button>
            );
          })}
        </div>
      )}

      {/* Main content: file tree (+ optional preview panel) */}
      {selectedFile ? (
        <Group orientation="vertical" className="flex-1 min-h-0">
          <Panel id="file-tree" defaultSize={45} minSize={20}>
            <div className="h-full overflow-auto v2-scrollbar">
              <FileTree
                files={files}
                onFileClick={handleFileClick}
                selectedPath={selectedFile.path}
              />
            </div>
          </Panel>
          <Separator
            className={cn(
              'h-[2px] bg-v2-border transition-colors duration-150',
              'hover:bg-v2-accent'
            )}
          />
          <Panel id="file-preview" defaultSize={55} minSize={25}>
            <FilePreview
              file={selectedFile}
              workspacePath={effectiveWorkspace}
              onOpenInTile={handleOpenInTile}
              onClose={() => setSelectedFile(null)}
            />
          </Panel>
        </Group>
      ) : (
        <div className="flex-1 overflow-auto v2-scrollbar">
          <FileTree files={files} onFileClick={handleFileClick} selectedPath={null} />
        </div>
      )}
    </div>
  );
}

// ============================================================================
// File Preview
// ============================================================================

interface FilePreviewProps {
  file: WorkspaceFileInfo;
  workspacePath: string;
  onOpenInTile: () => void;
  onClose: () => void;
}

function FilePreview({ file, workspacePath, onOpenInTile, onClose }: FilePreviewProps) {
  return (
    <div className="flex flex-col h-full bg-v2-surface">
      {/* Preview header */}
      <div className="flex items-center justify-between px-3 py-1.5 border-b border-v2-border shrink-0">
        <span className="text-xs text-v2-text-secondary truncate">{file.path}</span>
        <div className="flex items-center gap-1 shrink-0">
          <button
            onClick={onOpenInTile}
            className="p-1 text-v2-text-muted hover:text-v2-text transition-colors rounded hover:bg-v2-sidebar-hover"
            title="Open in new tile"
          >
            <ExternalLink className="w-3.5 h-3.5" />
          </button>
          <button
            onClick={onClose}
            className="p-1 text-v2-text-muted hover:text-v2-text transition-colors rounded hover:bg-v2-sidebar-hover"
            title="Close preview"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Content — rendered via InlineArtifactPreview */}
      <div className="flex-1 overflow-auto v2-scrollbar">
        <InlineArtifactPreview filePath={file.path} workspacePath={workspacePath} />
      </div>
    </div>
  );
}

// ============================================================================
// File Tree
// ============================================================================

interface FileTreeProps {
  files: WorkspaceFileInfo[];
  onFileClick: (file: WorkspaceFileInfo) => void;
  selectedPath: string | null;
}

function FileTree({ files, onFileClick, selectedPath }: FileTreeProps) {
  const tree = useMemo(() => buildTree(files), [files]);

  if (files.length === 0) {
    return (
      <div className="flex items-center justify-center h-full text-v2-text-muted text-sm">
        No files in workspace
      </div>
    );
  }

  return (
    <div className="py-1">
      {tree.children.map((node) => (
        <TreeNode
          key={node.path}
          node={node}
          depth={0}
          onFileClick={onFileClick}
          selectedPath={selectedPath}
        />
      ))}
    </div>
  );
}

// ============================================================================
// Tree Node
// ============================================================================

interface TreeNodeData {
  name: string;
  path: string;
  isDir: boolean;
  file?: WorkspaceFileInfo;
  children: TreeNodeData[];
}

interface TreeNodeProps {
  node: TreeNodeData;
  depth: number;
  onFileClick: (file: WorkspaceFileInfo) => void;
  selectedPath: string | null;
}

function TreeNode({ node, depth, onFileClick, selectedPath }: TreeNodeProps) {
  const [expanded, setExpanded] = useState(depth < 1);
  const isPreviewable = !!node.file && canPreviewFile(node.file.path);

  if (node.isDir) {
    return (
      <div>
        <button
          onClick={() => setExpanded(!expanded)}
          className={cn(
            'flex items-center gap-1.5 w-full text-sm text-v2-text-secondary',
            'hover:bg-v2-sidebar-hover hover:text-v2-text',
            'transition-colors duration-100 py-0.5 pr-2'
          )}
          style={{ paddingLeft: `${depth * 16 + 8}px` }}
        >
          <svg
            className={cn('w-3 h-3 shrink-0 transition-transform', expanded && 'rotate-90')}
            viewBox="0 0 12 12"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
          >
            <path d="M4 2l4 4-4 4" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <svg className="w-3.5 h-3.5 shrink-0 text-v2-text-muted" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
            <path d="M2 4.5A1.5 1.5 0 013.5 3h3l1.5 1.5h4.5A1.5 1.5 0 0114 6v5.5a1.5 1.5 0 01-1.5 1.5h-9A1.5 1.5 0 012 11.5V4.5z" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
          <span className="truncate">{node.name}</span>
        </button>
        {expanded &&
          node.children.map((child) => (
            <TreeNode
              key={child.path}
              node={child}
              depth={depth + 1}
              onFileClick={onFileClick}
              selectedPath={selectedPath}
            />
          ))}
      </div>
    );
  }

  const isSelected = selectedPath === node.path;

  return (
    <button
      onClick={() => node.file && onFileClick(node.file)}
      className={cn(
        'flex items-center gap-1.5 w-full text-sm text-v2-text-secondary',
        'hover:bg-v2-sidebar-hover hover:text-v2-text',
        'transition-colors duration-100 py-0.5 pr-2',
        isSelected && 'bg-[var(--v2-channel-active)] text-v2-text',
        isPreviewable && !isSelected && 'text-v2-text'
      )}
      style={{ paddingLeft: `${depth * 16 + 8 + 15}px` }}
    >
      <FileIcon name={node.name} />
      <span className="truncate">{node.name}</span>
      {isPreviewable && (
        <span
          aria-label="Rich preview available"
          title="Rich preview available"
          className="shrink-0 text-v2-accent"
        >
          <Eye className="w-3.5 h-3.5" />
        </span>
      )}
      {node.file && (
        <span className="ml-auto text-[10px] text-v2-text-muted shrink-0">
          {formatSize(node.file.size)}
        </span>
      )}
    </button>
  );
}

// ============================================================================
// Helpers
// ============================================================================

function buildTree(files: WorkspaceFileInfo[]): TreeNodeData {
  const root: TreeNodeData = { name: '', path: '', isDir: true, children: [] };

  for (const file of files) {
    const parts = file.path.split('/').filter(Boolean);
    let current = root;

    for (let i = 0; i < parts.length; i++) {
      const part = parts[i];
      const isLast = i === parts.length - 1;
      const path = parts.slice(0, i + 1).join('/');

      if (isLast) {
        current.children.push({
          name: part,
          path,
          isDir: false,
          file,
          children: [],
        });
      } else {
        let dir = current.children.find((c) => c.isDir && c.name === part);
        if (!dir) {
          dir = { name: part, path, isDir: true, children: [] };
          current.children.push(dir);
        }
        current = dir;
      }
    }
  }

  const sortChildren = (node: TreeNodeData) => {
    node.children.sort((a, b) => {
      if (a.isDir !== b.isDir) return a.isDir ? -1 : 1;
      return a.name.localeCompare(b.name);
    });
    node.children.forEach(sortChildren);
  };
  sortChildren(root);

  return root;
}

function formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes}B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)}K`;
  return `${(bytes / (1024 * 1024)).toFixed(1)}M`;
}

function findMainPreviewableFile(files: WorkspaceFileInfo[]): WorkspaceFileInfo | null {
  const previewableFiles = files.filter((file) => canPreviewFile(file.path));
  if (previewableFiles.length === 0) return null;

  const pdf = previewableFiles.find((file) =>
    file.path.toLowerCase().endsWith('.pdf')
  );
  if (pdf) return pdf;

  const pptx = previewableFiles.find((file) =>
    file.path.toLowerCase().endsWith('.pptx')
  );
  if (pptx) return pptx;

  const docx = previewableFiles.find((file) =>
    file.path.toLowerCase().endsWith('.docx')
  );
  if (docx) return docx;

  const indexHtml = previewableFiles.find((file) =>
    file.path.toLowerCase().endsWith('index.html')
  );
  if (indexHtml) return indexHtml;

  const anyHtml = previewableFiles.find((file) =>
    file.path.toLowerCase().endsWith('.html') ||
    file.path.toLowerCase().endsWith('.htm')
  );
  if (anyHtml) return anyHtml;

  const image = previewableFiles.find((file) => {
    const lower = file.path.toLowerCase();
    return (
      lower.endsWith('.png') ||
      lower.endsWith('.jpg') ||
      lower.endsWith('.jpeg') ||
      lower.endsWith('.gif') ||
      lower.endsWith('.svg') ||
      lower.endsWith('.webp')
    );
  });
  if (image) return image;

  return previewableFiles[0];
}

function FileIcon({ name }: { name: string }) {
  const ext = name.split('.').pop()?.toLowerCase();
  const isCode = ['ts', 'tsx', 'js', 'jsx', 'py', 'rs', 'go', 'java', 'cpp', 'c', 'h'].includes(ext || '');
  const isConfig = ['json', 'yaml', 'yml', 'toml', 'ini', 'env'].includes(ext || '');
  const isMarkdown = ext === 'md';

  const className = cn(
    'w-3.5 h-3.5 shrink-0',
    isCode ? 'text-blue-400' : isConfig ? 'text-amber-400' : isMarkdown ? 'text-emerald-400' : 'text-v2-text-muted'
  );

  return (
    <svg className={className} viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
      <path d="M4 2h5l3 3v9H4V2z" strokeLinejoin="round" />
      <path d="M9 2v3h3" strokeLinejoin="round" />
    </svg>
  );
}
