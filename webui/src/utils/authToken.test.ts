import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import {
  AUTH_TOKEN_QUERY_PARAM,
  AUTH_TOKEN_STORAGE_KEY,
  bootstrapAuthTokenFromUrl,
  buildAuthenticatedUrl,
  getStoredAuthToken,
  installAuthenticatedFetch,
} from './authToken';

describe('authToken utilities', () => {
  beforeEach(() => {
    sessionStorage.clear();
  });

  afterEach(() => {
    sessionStorage.clear();
    vi.unstubAllGlobals();
  });

  it('bootstraps token from URL and strips query token', () => {
    window.history.replaceState({}, '', `/?${AUTH_TOKEN_QUERY_PARAM}=abc123&v=2`);

    const token = bootstrapAuthTokenFromUrl();

    expect(token).toBe('abc123');
    expect(sessionStorage.getItem(AUTH_TOKEN_STORAGE_KEY)).toBe('abc123');
    expect(window.location.search).toBe('?v=2');
  });

  it('builds authenticated URLs when token exists', () => {
    sessionStorage.setItem(AUTH_TOKEN_STORAGE_KEY, 'abc123');

    expect(buildAuthenticatedUrl('/ws/session-1')).toContain('token=abc123');
    expect(buildAuthenticatedUrl('/api/sessions?x=1')).toContain('x=1');
    expect(buildAuthenticatedUrl('/api/sessions?x=1')).toContain('token=abc123');
  });

  it('installs fetch wrapper that injects bearer auth', async () => {
    sessionStorage.setItem(AUTH_TOKEN_STORAGE_KEY, 'abc123');
    const fetchMock = vi.fn().mockResolvedValue(new Response('{}', { status: 200 }));
    vi.stubGlobal('fetch', fetchMock);

    installAuthenticatedFetch();
    await fetch('/api/providers');

    const [, init] = fetchMock.mock.calls[0] as [RequestInfo | URL, RequestInit];
    const headers = new Headers(init.headers);
    expect(headers.get('Authorization')).toBe('Bearer abc123');
  });

  it('returns null when no token is stored', () => {
    expect(getStoredAuthToken()).toBeNull();
  });
});
