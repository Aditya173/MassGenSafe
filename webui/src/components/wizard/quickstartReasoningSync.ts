export type QuickstartReasoningEffort = 'low' | 'medium' | 'high' | 'xhigh' | 'max';

export interface QuickstartReasoningProfile {
  choices: Array<[string, QuickstartReasoningEffort]>;
  default_effort: QuickstartReasoningEffort;
  description: string;
}

export interface QuickstartReasoningSyncInput {
  profile: QuickstartReasoningProfile | null;
  profileKey: string | null;
  lastAppliedProfileKey?: string | null;
  currentEffort?: QuickstartReasoningEffort;
}

export interface QuickstartReasoningSyncDecision {
  nextEffort: QuickstartReasoningEffort | null;
  nextProfileKey: string | null;
  shouldApply: boolean;
}

export function buildQuickstartReasoningProfileKey(
  providerId?: string | null,
  model?: string | null,
): string | null {
  if (!providerId || !model) {
    return null;
  }
  return `${providerId}::${model}`;
}

export function resolveQuickstartReasoningSync({
  profile,
  profileKey,
  lastAppliedProfileKey,
  currentEffort,
}: QuickstartReasoningSyncInput): QuickstartReasoningSyncDecision {
  if (!profile || !profileKey) {
    return {
      nextEffort: null,
      nextProfileKey: null,
      shouldApply: currentEffort !== undefined,
    };
  }

  const validEfforts = new Set(profile.choices.map(([, value]) => value));
  const currentEffortIsValid = currentEffort !== undefined && validEfforts.has(currentEffort);
  const hasStoredProfileKey = lastAppliedProfileKey !== undefined;

  if (!hasStoredProfileKey && currentEffortIsValid) {
    return {
      nextEffort: currentEffort,
      nextProfileKey: profileKey,
      shouldApply: false,
    };
  }

  if (lastAppliedProfileKey !== profileKey || !currentEffortIsValid) {
    return {
      nextEffort: profile.default_effort,
      nextProfileKey: profileKey,
      shouldApply: currentEffort !== profile.default_effort,
    };
  }

  return {
    nextEffort: currentEffort,
    nextProfileKey: profileKey,
    shouldApply: false,
  };
}
