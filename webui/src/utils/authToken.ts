export const AUTH_TOKEN_STORAGE_KEY = 'massgen_auth_token';
export const AUTH_TOKEN_QUERY_PARAM = 'token';

function getWindowUrl(rawUrl: string): URL {
  return new URL(rawUrl, window.location.origin);
}

export function getStoredAuthToken(): string | null {
  const value = sessionStorage.getItem(AUTH_TOKEN_STORAGE_KEY);
  return value && value.trim().length > 0 ? value.trim() : null;
}

export function setStoredAuthToken(token: string): void {
  sessionStorage.setItem(AUTH_TOKEN_STORAGE_KEY, token);
}

export function bootstrapAuthTokenFromUrl(): string | null {
  const url = getWindowUrl(window.location.href);
  const token = url.searchParams.get(AUTH_TOKEN_QUERY_PARAM);
  if (!token) {
    return getStoredAuthToken();
  }

  setStoredAuthToken(token);
  url.searchParams.delete(AUTH_TOKEN_QUERY_PARAM);
  const search = url.searchParams.toString();
  const rewritten = `${url.pathname}${search ? `?${search}` : ''}${url.hash}`;
  window.history.replaceState({}, '', rewritten);
  return token;
}

export function buildAuthenticatedUrl(rawUrl: string): string {
  const token = getStoredAuthToken();
  if (!token) {
    return rawUrl;
  }

  const url = getWindowUrl(rawUrl);
  if (!url.searchParams.has(AUTH_TOKEN_QUERY_PARAM)) {
    url.searchParams.set(AUTH_TOKEN_QUERY_PARAM, token);
  }

  const isAbsolute = /^(https?:|wss?:)/i.test(rawUrl);
  if (isAbsolute) {
    return url.toString();
  }
  return `${url.pathname}${url.search}${url.hash}`;
}

export function buildAuthHeaders(existing?: HeadersInit): Headers {
  const headers = new Headers(existing);
  const token = getStoredAuthToken();
  if (token && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${token}`);
  }
  return headers;
}

let fetchWrapped = false;

export function installAuthenticatedFetch(): void {
  if (fetchWrapped) {
    return;
  }
  fetchWrapped = true;

  const originalFetch = window.fetch.bind(window);
  window.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
    const token = getStoredAuthToken();
    if (!token) {
      return originalFetch(input, init);
    }

    const nextInit: RequestInit = {
      ...(init || {}),
      headers: buildAuthHeaders(init?.headers),
    };
    return originalFetch(input, nextInit);
  }) as typeof window.fetch;
}
