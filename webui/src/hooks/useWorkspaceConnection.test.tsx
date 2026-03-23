import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { useAgentStore } from '../stores/agentStore'
import { useWorkspaceStore } from '../stores/workspaceStore'
import { useWorkspaceConnection } from './useWorkspaceConnection'

class MockWebSocket {
  static CONNECTING = 0
  static OPEN = 1
  static CLOSING = 2
  static CLOSED = 3
  static instances: MockWebSocket[] = []

  url: string
  readyState = MockWebSocket.CONNECTING
  sent: string[] = []
  onopen: ((event: Event) => void) | null = null
  onmessage: ((event: MessageEvent) => void) | null = null
  onclose: ((event: CloseEvent) => void) | null = null
  onerror: ((event: Event) => void) | null = null

  constructor(url: string) {
    this.url = url
    MockWebSocket.instances.push(this)
  }

  send(data: string) {
    this.sent.push(data)
  }

  close(code = 1000, reason = 'Client disconnect') {
    this.readyState = MockWebSocket.CLOSED
    this.onclose?.({ code, reason } as CloseEvent)
  }

  open() {
    this.readyState = MockWebSocket.OPEN
    this.onopen?.(new Event('open'))
  }

  receive(data: unknown) {
    this.onmessage?.({ data: JSON.stringify(data) } as MessageEvent)
  }
}

function sentMessages(socket: MockWebSocket): Array<Record<string, unknown>> {
  return socket.sent.map((message) => JSON.parse(message) as Record<string, unknown>)
}

describe('useWorkspaceConnection', () => {
  beforeEach(() => {
    vi.useFakeTimers()
    MockWebSocket.instances = []
    vi.stubGlobal('WebSocket', MockWebSocket)
    useAgentStore.getState().reset()
    useWorkspaceStore.getState().reset()
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.unstubAllGlobals()
    useAgentStore.getState().reset()
    useWorkspaceStore.getState().reset()
  })

  it('keeps polling watch_session until the workspace receives files', () => {
    act(() => {
      useAgentStore.setState({ sessionId: 'session-1', isComplete: false })
    })

    const { unmount } = renderHook(() => useWorkspaceConnection())
    const socket = MockWebSocket.instances[0]

    expect(socket).toBeDefined()

    act(() => {
      socket.open()
    })

    expect(sentMessages(socket)).toEqual([{ action: 'watch_session' }])

    act(() => {
      socket.receive({
        type: 'workspace_connected',
        initial_files: {
          '/tmp/workspace1': [],
        },
      })
    })

    expect(useWorkspaceStore.getState().getWorkspaceFiles('/tmp/workspace1')).toEqual([])

    act(() => {
      vi.advanceTimersByTime(5000)
    })

    expect(sentMessages(socket)).toEqual([
      { action: 'watch_session' },
      { action: 'watch_session' },
    ])

    act(() => {
      socket.receive({
        type: 'workspace_connected',
        initial_files: {
          '/tmp/workspace1': [
            {
              path: 'poem_about_love.txt',
              size: 123,
              modified: 1700000000,
            },
          ],
        },
      })
    })

    act(() => {
      vi.advanceTimersByTime(5000)
    })

    expect(sentMessages(socket)).toEqual([
      { action: 'watch_session' },
      { action: 'watch_session' },
    ])

    unmount()
  })

  it('uses the backend refresh payload key expected by the websocket endpoint', () => {
    act(() => {
      useAgentStore.setState({ sessionId: 'session-2', isComplete: false })
    })

    const { result, unmount } = renderHook(() => useWorkspaceConnection())
    const socket = MockWebSocket.instances[0]

    expect(socket).toBeDefined()

    act(() => {
      socket.open()
    })

    act(() => {
      result.current.requestRefresh('/tmp/workspace1')
    })

    const messages = sentMessages(socket)

    expect(messages[messages.length - 1]).toEqual({
      action: 'refresh',
      path: '/tmp/workspace1',
    })

    unmount()
  })

  it('keeps polling briefly after completion when the workspace is still empty', () => {
    act(() => {
      useAgentStore.setState({ sessionId: 'session-3', isComplete: false })
    })

    const { unmount } = renderHook(() => useWorkspaceConnection())
    const socket = MockWebSocket.instances[0]

    expect(socket).toBeDefined()

    act(() => {
      socket.open()
    })

    act(() => {
      socket.receive({
        type: 'workspace_connected',
        initial_files: {
          '/tmp/workspace1': [],
        },
      })
    })

    act(() => {
      useAgentStore.setState({ isComplete: true })
    })

    act(() => {
      vi.advanceTimersByTime(5000)
    })

    expect(sentMessages(socket)).toEqual([
      { action: 'watch_session' },
      { action: 'watch_session' },
    ])

    unmount()
  })
})
