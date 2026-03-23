import { fireEvent, render, screen } from '@testing-library/react'
import { beforeEach, describe, expect, it } from 'vitest'
import { useMessageStore } from '../../../stores/v2/messageStore'
import { TaskPlanPanel } from './TaskPlanPanel'

describe('TaskPlanPanel', () => {
  beforeEach(() => {
    useMessageStore.getState().reset()
    useMessageStore.setState({
      taskPlan: [
        {
          id: 'task-1',
          description: 'Build the first artifact',
          status: 'in_progress',
          priority: 'high',
        },
        {
          id: 'task-2',
          description: 'Verify the result',
          status: 'pending',
        },
      ],
    })
  })

  it('renders docked at default width and supports drag-resize', () => {
    render(<TaskPlanPanel />)

    const panel = screen.getByTestId('task-plan-panel')
    expect(panel).toHaveStyle({ width: '360px' })

    const handle = screen.getByTestId('task-plan-resize-handle')
    fireEvent.mouseDown(handle, { clientX: 100 })
    fireEvent.mouseMove(window, { clientX: 40 })
    fireEvent.mouseUp(window)

    expect(panel).toHaveStyle({ width: '420px' })
  })

  it('clicking a task expands its metadata', () => {
    render(<TaskPlanPanel />)

    // Click first task
    fireEvent.click(screen.getByText(/Build the first artifact/))

    // Should show metadata
    expect(screen.getByText('in progress')).toBeInTheDocument()
    expect(screen.getByText('high')).toBeInTheDocument()
    expect(screen.getByText('task-1')).toBeInTheDocument()
  })

  it('collapses to a vertical icon strip and re-expands', () => {
    render(<TaskPlanPanel />)

    // Find the collapse button (the chevron in header)
    const header = screen.getByTestId('task-plan-panel').querySelector('button:last-child')
    if (header) fireEvent.click(header)

    // Panel should be gone, collapsed strip visible
    expect(screen.queryByTestId('task-plan-panel')).toBeNull()
    expect(screen.getByText('0/2')).toBeInTheDocument()

    // Re-expand
    fireEvent.click(screen.getByTitle(/click to expand/))
    expect(screen.getByTestId('task-plan-panel')).toBeInTheDocument()
  })

  it('returns null when no task plan exists', () => {
    useMessageStore.setState({ taskPlan: [] })
    const { container } = render(<TaskPlanPanel />)
    expect(container.innerHTML).toBe('')
  })
})
