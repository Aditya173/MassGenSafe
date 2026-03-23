import { act } from 'react'
import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { useAgentStore } from '../../../stores/agentStore'
import { useWorkspaceStore } from '../../../stores/workspaceStore'
import { WorkspaceBrowserTile } from './WorkspaceBrowserTile'

vi.mock('react-resizable-panels', () => ({
  Group: ({
    orientation,
    className,
    children,
  }: {
    orientation: 'horizontal' | 'vertical'
    className?: string
    children: React.ReactNode
  }) => (
    <div data-testid="panel-group" data-orientation={orientation} className={className}>
      {children}
    </div>
  ),
  Panel: ({ children }: { children: React.ReactNode }) => <div>{children}</div>,
  Separator: ({ className }: { className?: string }) => (
    <div data-testid="panel-separator" className={className} />
  ),
}))

vi.mock('../../InlineArtifactPreview', () => ({
  InlineArtifactPreview: ({ filePath }: { filePath: string }) => (
    <div data-testid="inline-artifact-preview">{filePath}</div>
  ),
}))

describe('WorkspaceBrowserTile', () => {
  beforeEach(() => {
    useAgentStore.getState().reset()
    useWorkspaceStore.getState().reset()

    act(() => {
      useAgentStore.getState().initSession('session-1', 'Inspect files', ['agent_a'], 'dark')
      useWorkspaceStore.getState().setInitialFiles('/tmp/workspace', [
        {
          path: 'tasks/plan.json',
          size: 5300,
          modified: 1,
        },
        {
          path: 'deliverables/index.html',
          size: 2400,
          modified: 2,
        },
      ])
    })
  })

  it('auto-previews the main artifact and marks previewable files in the tree', () => {
    render(<WorkspaceBrowserTile />)

    expect(screen.getByTestId('inline-artifact-preview')).toHaveTextContent('deliverables/index.html')
    expect(screen.getAllByLabelText('Rich preview available')).toHaveLength(1)
  })

  it('stacks the inline preview below the file tree after selecting a file', () => {
    render(<WorkspaceBrowserTile />)

    fireEvent.click(screen.getByText('plan.json'))

    expect(screen.getByTestId('inline-artifact-preview')).toHaveTextContent('tasks/plan.json')
    expect(screen.getByTestId('panel-group')).toHaveAttribute('data-orientation', 'vertical')
  })
})
