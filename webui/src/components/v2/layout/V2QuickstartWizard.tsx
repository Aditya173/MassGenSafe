/**
 * V2 Quickstart Wizard Overlay
 *
 * Full-screen overlay for guided config setup.
 * Reuses existing wizard step components and the wizardStore for all state.
 */

import { useCallback, useState } from 'react';
import { cn } from '../../../lib/utils';
import { useWizardStore, WizardStep } from '../../../stores/wizardStore';
import {
  DockerStep,
  ApiKeyStep,
  AgentCountStep,
  SetupModeStep,
  AgentConfigStep,
  CoordinationStep,
  PreviewStep,
} from '../../wizard';

const stepConfig: Record<WizardStep, { title: string }> = {
  docker: { title: 'Execution Mode' },
  apiKeys: { title: 'API Keys' },
  agentCount: { title: 'Number of Agents' },
  setupMode: { title: 'Setup Mode' },
  agentConfig: { title: 'Agent Configuration' },
  coordination: { title: 'Coordination Settings' },
  preview: { title: 'Review & Save' },
};

interface V2QuickstartWizardProps {
  onConfigSaved?: (configPath: string) => void;
  temporaryMode?: boolean;
}

export function V2QuickstartWizard({ onConfigSaved, temporaryMode = false }: V2QuickstartWizardProps) {
  const isOpen = useWizardStore((s) => s.isOpen);
  const currentStep = useWizardStore((s) => s.currentStep);
  const isLoading = useWizardStore((s) => s.isLoading);
  const agents = useWizardStore((s) => s.agents);
  const agentCount = useWizardStore((s) => s.agentCount);
  const providers = useWizardStore((s) => s.providers);

  const closeWizard = useWizardStore((s) => s.closeWizard);
  const nextStep = useWizardStore((s) => s.nextStep);
  const prevStep = useWizardStore((s) => s.prevStep);
  const saveConfig = useWizardStore((s) => s.saveConfig);
  const reset = useWizardStore((s) => s.reset);

  const [temporarySessionState, setTemporarySessionState] = useState<'idle' | 'completed' | 'cancelled'>('idle');
  const [temporarySessionPending, setTemporarySessionPending] = useState(false);
  const [temporarySessionError, setTemporarySessionError] = useState<string | null>(null);

  const attemptWindowClose = useCallback(() => {
    window.close();
  }, []);

  const finalizeTemporarySession = useCallback(
    async (status: 'completed' | 'cancelled', configPath?: string | null) => {
      setTemporarySessionPending(true);
      setTemporarySessionError(null);

      try {
        const response = await fetch(
          status === 'completed' ? '/api/quickstart/complete' : '/api/quickstart/cancel',
          {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: status === 'completed'
              ? JSON.stringify({ config_path: configPath ?? null })
              : undefined,
          },
        );

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(
            errorData.error || `Failed to ${status === 'completed' ? 'finish' : 'cancel'} session`,
          );
        }

        if (status === 'completed' && configPath && onConfigSaved) {
          onConfigSaved(configPath);
        }

        setTemporarySessionState(status);
        attemptWindowClose();
      } catch (err) {
        setTemporarySessionError(err instanceof Error ? err.message : 'Unexpected error');
      } finally {
        setTemporarySessionPending(false);
      }
    },
    [attemptWindowClose, onConfigSaved],
  );

  const handleClose = useCallback(() => {
    if (temporaryMode && temporarySessionState === 'idle') {
      void finalizeTemporarySession('cancelled');
      return;
    }

    closeWizard();
    reset();
  }, [closeWizard, finalizeTemporarySession, reset, temporaryMode, temporarySessionState]);

  const handleSave = useCallback(async () => {
    const success = await saveConfig();
    if (success) {
      const configPath = useWizardStore.getState().savedConfigPath;

      if (temporaryMode) {
        await finalizeTemporarySession('completed', configPath);
        return;
      }

      if (configPath && onConfigSaved) {
        onConfigSaved(configPath);
      }

      closeWizard();
      reset();
    }
  }, [closeWizard, finalizeTemporarySession, onConfigSaved, reset, saveConfig, temporaryMode]);

  // Compute visible steps (skip steps that don't apply)
  const visibleSteps = (
    ['docker', 'apiKeys', 'agentCount', 'setupMode', 'agentConfig', 'coordination', 'preview'] as WizardStep[]
  ).filter((step) => {
    if (step === 'apiKeys' && providers.some((p) => p.has_api_key)) return false;
    if (step === 'setupMode' && agentCount === 1) return false;
    if (step === 'coordination' && agentCount === 1) return false;
    return true;
  });

  const currentStepIndex = visibleSteps.indexOf(currentStep);

  const canProceed = useCallback(() => {
    switch (currentStep) {
      case 'docker':
        return true;
      case 'apiKeys':
        return providers.some((p) => p.has_api_key);
      case 'agentCount':
        return agentCount >= 1 && agentCount <= 5;
      case 'setupMode':
        return true;
      case 'agentConfig':
        return agents.every((agent) => agent.provider && agent.model);
      case 'coordination':
        return true;
      case 'preview':
        return true;
      default:
        return false;
    }
  }, [currentStep, agentCount, agents, providers]);

  const renderStep = () => {
    switch (currentStep) {
      case 'docker': return <DockerStep />;
      case 'apiKeys': return <ApiKeyStep />;
      case 'agentCount': return <AgentCountStep />;
      case 'setupMode': return <SetupModeStep />;
      case 'agentConfig': return <AgentConfigStep />;
      case 'coordination': return <CoordinationStep />;
      case 'preview': return <PreviewStep />;
      default: return null;
    }
  };

  if (!isOpen) return null;

  const stepInfo = stepConfig[currentStep];
  const isFirstStep = currentStep === 'docker';
  const isLastStep = currentStep === 'preview';
  const showTemporaryResult = temporarySessionState !== 'idle';

  return (
    <div className="fixed inset-0 z-50 bg-v2-main flex flex-col animate-v2-overlay-backdrop">
      <div className="flex flex-col h-full animate-v2-overlay-content">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-v2-border bg-v2-surface shrink-0">
          <div className="flex items-center gap-3">
            {/* Wand icon */}
            <div className="p-2 bg-v2-accent/10 rounded-lg">
              <svg
                width="20" height="20" viewBox="0 0 24 24"
                fill="none" stroke="currentColor" strokeWidth="2"
                strokeLinecap="round" strokeLinejoin="round"
                className="text-v2-accent"
              >
                <path d="M15 4V2" /><path d="M15 16v-2" /><path d="M8 9h2" />
                <path d="M20 9h2" /><path d="M17.8 11.8L19 13" />
                <path d="M15 9h0" /><path d="M17.8 6.2L19 5" />
                <path d="M3 21l9-9" /><path d="M12.2 6.2L11 5" />
              </svg>
            </div>
            <div>
              <h2 className="text-lg font-semibold text-v2-text">
                {showTemporaryResult ? 'Setup Complete' : 'Quickstart Setup'}
              </h2>
              <p className="text-xs text-v2-text-muted">
                {showTemporaryResult
                  ? 'Temporary session'
                  : `Step ${currentStepIndex + 1} of ${visibleSteps.length} \u2014 ${stepInfo.title}`}
              </p>
            </div>
          </div>

          <button
            onClick={handleClose}
            disabled={isLoading || temporarySessionPending}
            title="Close wizard"
            className={cn(
              'p-2 rounded-v2-input',
              'text-v2-text-secondary hover:text-v2-text',
              'hover:bg-v2-sidebar-hover',
              'transition-colors duration-150',
              'disabled:opacity-50 disabled:cursor-not-allowed'
            )}
          >
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path d="M4 4l8 8M12 4l-8 8" strokeLinecap="round" />
            </svg>
          </button>
        </div>

        {/* Progress dots */}
        {!showTemporaryResult && (
          <div className="px-6 py-2 bg-v2-surface border-b border-v2-border" data-testid="progress-dots">
            <div className="flex items-center gap-2">
              {visibleSteps.map((step, index) => {
                const isActive = index === currentStepIndex;
                const isComplete = index < currentStepIndex;
                return (
                  <div key={step} className="flex-1" data-testid="progress-dot">
                    <div
                      className={cn(
                        'h-1.5 rounded-full transition-colors duration-200',
                        isComplete ? 'bg-v2-accent' :
                        isActive ? 'bg-v2-accent/50' :
                        'bg-v2-border'
                      )}
                    />
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Content */}
        <div className="flex-1 overflow-y-auto v2-scrollbar">
          <div className="max-w-4xl mx-auto px-6 py-8">
            {showTemporaryResult ? (
              <div className="flex flex-col items-center text-center py-12">
                <div className={cn(
                  'mb-5 flex h-16 w-16 items-center justify-center rounded-full',
                  temporarySessionState === 'completed'
                    ? 'bg-green-500/10 text-green-400'
                    : 'bg-amber-500/10 text-amber-400'
                )}>
                  {temporarySessionState === 'completed' ? (
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M20 6L9 17l-5-5" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                  ) : (
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <circle cx="12" cy="12" r="10" /><path d="M15 9l-6 6M9 9l6 6" strokeLinecap="round" />
                    </svg>
                  )}
                </div>
                <h3 className="text-xl font-semibold text-v2-text">
                  {temporarySessionState === 'completed' ? 'Setup Complete' : 'Setup Cancelled'}
                </h3>
                <p className="mt-3 text-sm text-v2-text-secondary">
                  {temporarySessionState === 'completed'
                    ? 'This tab can be closed.'
                    : 'The session has been cancelled. This tab can be closed.'}
                </p>
                {temporarySessionError && (
                  <div className="mt-4 w-full max-w-md rounded-lg border border-red-500/20 bg-red-500/5 px-4 py-3 text-sm text-red-400">
                    {temporarySessionError}
                  </div>
                )}
                <button
                  onClick={attemptWindowClose}
                  className={cn(
                    'mt-6 rounded-v2-input px-5 py-2.5 text-sm font-medium',
                    'bg-v2-accent text-white hover:bg-v2-accent-hover',
                    'transition-colors duration-150'
                  )}
                >
                  Close Window
                </button>
              </div>
            ) : (
              renderStep()
            )}
          </div>
        </div>

        {/* Footer */}
        {!showTemporaryResult && (
          <div className="flex items-center justify-between px-6 py-4 border-t border-v2-border bg-v2-surface shrink-0">
            <button
              onClick={isFirstStep ? handleClose : prevStep}
              disabled={isLoading || temporarySessionPending}
              className={cn(
                'flex items-center gap-2 text-sm px-3 py-2 rounded-v2-input',
                'text-v2-text-secondary hover:text-v2-text',
                'transition-colors duration-150',
                'disabled:opacity-50 disabled:cursor-not-allowed'
              )}
            >
              <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path d="M10 4l-4 4 4 4" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
              {isFirstStep ? 'Cancel' : 'Back'}
            </button>

            {isLastStep ? (
              <button
                onClick={handleSave}
                disabled={isLoading || temporarySessionPending || !canProceed()}
                className={cn(
                  'flex items-center gap-2 rounded-v2-input px-5 py-2.5 text-sm font-medium',
                  'bg-green-600 text-white hover:bg-green-500',
                  'disabled:opacity-40 disabled:cursor-not-allowed',
                  'transition-colors duration-150'
                )}
              >
                {isLoading || temporarySessionPending ? (
                  <>
                    <span className="w-3.5 h-3.5 rounded-full border-2 border-white/30 border-t-white animate-spin" />
                    {temporaryMode ? 'Finishing...' : 'Saving...'}
                  </>
                ) : (
                  <>
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M15 4V2M15 16v-2M8 9h2M20 9h2M17.8 11.8L19 13M15 9h0M17.8 6.2L19 5M3 21l9-9M12.2 6.2L11 5" strokeLinecap="round" strokeLinejoin="round" />
                    </svg>
                    {temporaryMode ? 'Save & Finish' : 'Save & Start'}
                  </>
                )}
              </button>
            ) : (
              <button
                onClick={nextStep}
                disabled={!canProceed() || temporarySessionPending}
                className={cn(
                  'flex items-center gap-2 rounded-v2-input px-5 py-2.5 text-sm font-medium',
                  'bg-v2-accent text-white hover:bg-v2-accent-hover',
                  'disabled:opacity-40 disabled:cursor-not-allowed',
                  'transition-colors duration-150'
                )}
              >
                Next
                <svg width="14" height="14" viewBox="0 0 16 16" fill="none" stroke="currentColor" strokeWidth="1.5">
                  <path d="M6 4l4 4-4 4" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
