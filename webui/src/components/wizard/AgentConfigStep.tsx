/**
 * Agent Config Step Component
 *
 * Fourth step - configure provider and model for each agent.
 * Full-page scrollable layout with large, visible cards for easy selection.
 */

import { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion';
import { Bot, AlertCircle, Check, Search, Sparkles, X, Globe, Code, MessageSquare } from 'lucide-react';
import { useWizardStore, ProviderInfo, ProviderCapabilities, ReasoningEffort } from '../../stores/wizardStore';
import {
  buildQuickstartReasoningProfileKey,
  resolveQuickstartReasoningSync,
  type QuickstartReasoningProfile,
} from './quickstartReasoningSync';

interface ProviderCardProps {
  provider: ProviderInfo;
  isSelected: boolean;
  onSelect: () => void;
  disabled?: boolean;
}

const AGENT_FRAMEWORK_PROVIDER_IDS = new Set(['claude_code', 'codex', 'copilot']);

function isAgentFrameworkProvider(provider: ProviderInfo): boolean {
  return provider.is_agent_framework || AGENT_FRAMEWORK_PROVIDER_IDS.has(provider.id);
}

function ProviderCard({ provider, isSelected, onSelect, disabled }: ProviderCardProps) {
  const isAgentFramework = isAgentFrameworkProvider(provider);

  return (
    <button
      onClick={onSelect}
      disabled={disabled}
      className={`w-full p-4 rounded-xl border-2 text-left transition-all ${
        disabled
          ? 'opacity-50 cursor-not-allowed border-gray-300 dark:border-gray-700 bg-gray-100 dark:bg-gray-800/50'
          : isSelected
          ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/30 ring-2 ring-blue-500/20'
          : 'border-gray-200 dark:border-gray-700 hover:border-blue-300 dark:hover:border-blue-600 hover:bg-gray-50 dark:hover:bg-gray-800'
      }`}
    >
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-2">
          <div className="font-semibold text-gray-800 dark:text-gray-200">
            {provider.name}
          </div>
          {isAgentFramework && (
            <div className="inline-flex items-center gap-1 rounded-full bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-800 dark:bg-amber-900/40 dark:text-amber-200">
              <Bot className="h-3 w-3" />
              Agent
            </div>
          )}
        </div>
        {isSelected && (
          <div className="flex-shrink-0 w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center">
            <Check className="w-4 h-4 text-white" />
          </div>
        )}
      </div>

      {disabled && (
        <div className="mt-1 text-xs text-red-500">
          Needs {provider.env_var}
        </div>
      )}
    </button>
  );
}

interface ModelCardProps {
  model: string;
  isSelected: boolean;
  isDefault: boolean;
  onSelect: () => void;
}

function ModelCard({ model, isSelected, isDefault, onSelect }: ModelCardProps) {
  return (
    <button
      onClick={onSelect}
      className={`w-full p-3 rounded-lg border-2 text-left transition-all ${
        isSelected
          ? 'border-green-500 bg-green-50 dark:bg-green-900/30 ring-2 ring-green-500/20'
          : 'border-gray-200 dark:border-gray-700 hover:border-green-300 dark:hover:border-green-600 hover:bg-gray-50 dark:hover:bg-gray-800'
      }`}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="font-medium text-gray-800 dark:text-gray-200 text-sm">
            {model}
          </span>
          {isDefault && (
            <span className="inline-flex items-center gap-1 text-xs px-2 py-0.5 bg-green-100 dark:bg-green-900/40 text-green-700 dark:text-green-300 rounded-full">
              <Sparkles className="w-3 h-3" />
              Recommended
            </span>
          )}
        </div>
        {isSelected && (
          <div className="flex-shrink-0 w-5 h-5 bg-green-500 rounded-full flex items-center justify-center">
            <Check className="w-3 h-3 text-white" />
          </div>
        )}
      </div>
    </button>
  );
}

// Per-agent option toggle component
interface OptionToggleProps {
  enabled: boolean;
  onChange: (enabled: boolean) => void;
  icon: React.ReactNode;
  label: string;
  description?: string;
}

function OptionToggle({ enabled, onChange, icon, label, description }: OptionToggleProps) {
  return (
    <label className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-800/50 rounded-lg cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors">
      <div className="flex items-center gap-3">
        <div className="text-gray-500 dark:text-gray-400">{icon}</div>
        <div>
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">{label}</span>
          {description && (
            <p className="text-xs text-gray-500 dark:text-gray-400">{description}</p>
          )}
        </div>
      </div>
      <div className="relative">
        <input
          type="checkbox"
          checked={enabled}
          onChange={(e) => onChange(e.target.checked)}
          className="sr-only peer"
        />
        <div className="w-10 h-6 bg-gray-200 dark:bg-gray-700 rounded-full peer-checked:bg-blue-500 transition-colors" />
        <div className="absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full shadow peer-checked:translate-x-4 transition-transform" />
      </div>
    </label>
  );
}

export function AgentConfigStep() {
  const providers = useWizardStore((s) => s.providers);
  const agents = useWizardStore((s) => s.agents);
  const setupMode = useWizardStore((s) => s.setupMode);
  const useDocker = useWizardStore((s) => s.useDocker);
  const setAgentConfig = useWizardStore((s) => s.setAgentConfig);
  const setAllAgentsConfig = useWizardStore((s) => s.setAllAgentsConfig);
  const setAgentReasoningEffort = useWizardStore((s) => s.setAgentReasoningEffort);
  const setAllAgentsReasoningEffort = useWizardStore((s) => s.setAllAgentsReasoningEffort);
  const setAgentWebSearch = useWizardStore((s) => s.setAgentWebSearch);
  const setAgentCodeExecution = useWizardStore((s) => s.setAgentCodeExecution);
  const setAgentSystemMessage = useWizardStore((s) => s.setAgentSystemMessage);
  const dynamicModels = useWizardStore((s) => s.dynamicModels);
  const loadingModels = useWizardStore((s) => s.loadingModels);
  const fetchDynamicModels = useWizardStore((s) => s.fetchDynamicModels);
  const providerCapabilities = useWizardStore((s) => s.providerCapabilities);
  const loadingCapabilities = useWizardStore((s) => s.loadingCapabilities);
  const fetchProviderCapabilities = useWizardStore((s) => s.fetchProviderCapabilities);

  // Search/filter state
  const [providerSearch, setProviderSearch] = useState('');
  const [modelSearch, setModelSearch] = useState('');
  const [reasoningProfile, setReasoningProfile] = useState<QuickstartReasoningProfile | null>(null);
  const [resolvedReasoningProfileKey, setResolvedReasoningProfileKey] = useState<string | null>(null);
  const [loadingReasoningProfile, setLoadingReasoningProfile] = useState(false);
  const lastAppliedReasoningProfileKeysRef = useRef<Record<string, string | null | undefined>>({});

  // System message mode for "same" setup mode: 'skip' | 'same' | 'different'
  const [systemMessageMode, setSystemMessageMode] = useState<'skip' | 'same' | 'different'>('skip');
  // Track which agent we're editing system message for (in 'different' mode)
  const [editingSystemMsgAgent, setEditingSystemMsgAgent] = useState<number | null>(null);

  // For multi-agent different config, track which agent we're configuring
  const [activeAgentIndex, setActiveAgentIndex] = useState(0);

  const availableProviders = providers.filter((p) => p.has_api_key);
  const unavailableProviders = providers.filter((p) => !p.has_api_key);

  // Get current agent's config
  const currentAgent = setupMode === 'same' ? agents[0] : agents[activeAgentIndex];
  const selectedProvider = providers.find((p) => p.id === currentAgent?.provider);

  // Fetch models when provider changes
  useEffect(() => {
    if (currentAgent?.provider && !dynamicModels[currentAgent.provider]) {
      fetchDynamicModels(currentAgent.provider);
    }
  }, [currentAgent?.provider, dynamicModels, fetchDynamicModels]);

  // Fetch capabilities when provider changes
  useEffect(() => {
    if (currentAgent?.provider && !providerCapabilities[currentAgent.provider]) {
      fetchProviderCapabilities(currentAgent.provider);
    }
  }, [currentAgent?.provider, providerCapabilities, fetchProviderCapabilities]);

  // Fetch quickstart reasoning profile when provider/model changes.
  useEffect(() => {
    const providerId = currentAgent?.provider;
    const model = currentAgent?.model;
    const requestProfileKey = buildQuickstartReasoningProfileKey(providerId, model);

    if (!providerId || !model) {
      setReasoningProfile(null);
      setResolvedReasoningProfileKey(null);
      setLoadingReasoningProfile(false);
      return;
    }

    let cancelled = false;
    setLoadingReasoningProfile(true);

    void fetch(
      `/api/quickstart/reasoning-profile?provider_id=${encodeURIComponent(providerId)}&model=${encodeURIComponent(model)}`,
    )
      .then(async (response) => {
        if (!response.ok) {
          throw new Error('Failed to fetch reasoning profile');
        }
        return response.json();
      })
      .then((data: { profile: QuickstartReasoningProfile | null }) => {
        if (!cancelled) {
          setReasoningProfile(data.profile);
          setResolvedReasoningProfileKey(requestProfileKey);
        }
      })
      .catch(() => {
        if (!cancelled) {
          setReasoningProfile(null);
          setResolvedReasoningProfileKey(requestProfileKey);
        }
      })
      .finally(() => {
        if (!cancelled) {
          setLoadingReasoningProfile(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, [currentAgent?.provider, currentAgent?.model]);

  // Keep reasoning selection aligned with the shared quickstart profile.
  useEffect(() => {
    if (!currentAgent) {
      return;
    }

    const applyReasoningEffort = (value?: ReasoningEffort) => {
      if (setupMode === 'same') {
        setAllAgentsReasoningEffort(value);
      } else {
        setAgentReasoningEffort(activeAgentIndex, value);
      }
    };

    const reasoningTargetKey = setupMode === 'same' ? 'all' : String(activeAgentIndex);
    const profileKey = buildQuickstartReasoningProfileKey(currentAgent.provider, currentAgent.model);
    if (profileKey && resolvedReasoningProfileKey !== profileKey) {
      return;
    }
    const decision = resolveQuickstartReasoningSync({
      profile: reasoningProfile,
      profileKey,
      lastAppliedProfileKey: lastAppliedReasoningProfileKeysRef.current[reasoningTargetKey],
      currentEffort: currentAgent.reasoning_effort,
    });

    lastAppliedReasoningProfileKeysRef.current[reasoningTargetKey] = decision.nextProfileKey;
    if (decision.shouldApply) {
      applyReasoningEffort(decision.nextEffort ?? undefined);
    }
  }, [
    activeAgentIndex,
    currentAgent,
    resolvedReasoningProfileKey,
    reasoningProfile,
    setAgentReasoningEffort,
    setAllAgentsReasoningEffort,
    setupMode,
  ]);

  // Get current provider's capabilities
  const currentCapabilities: ProviderCapabilities | null = currentAgent?.provider
    ? providerCapabilities[currentAgent.provider] || null
    : null;
  const isLoadingCapabilities = currentAgent?.provider
    ? loadingCapabilities[currentAgent.provider] || false
    : false;

  // Get available models for selected provider
  const availableModels = currentAgent?.provider
    ? (dynamicModels[currentAgent.provider] || selectedProvider?.models || []).filter(m => m !== 'custom')
    : [];

  // Filter providers by search
  const filteredAvailableProviders = providerSearch
    ? availableProviders.filter(p =>
        p.name.toLowerCase().includes(providerSearch.toLowerCase()) ||
        p.id.toLowerCase().includes(providerSearch.toLowerCase())
      )
    : availableProviders;

  const filteredUnavailableProviders = providerSearch
    ? unavailableProviders.filter(p =>
        p.name.toLowerCase().includes(providerSearch.toLowerCase()) ||
        p.id.toLowerCase().includes(providerSearch.toLowerCase())
      )
    : unavailableProviders;

  // Filter models by search
  const filteredModels = modelSearch
    ? availableModels.filter(m => m.toLowerCase().includes(modelSearch.toLowerCase()))
    : availableModels;

  // Handle provider selection
  const handleProviderSelect = (providerId: string) => {
    const providerInfo = providers.find((p) => p.id === providerId);
    const defaultModel = providerInfo?.default_model === 'custom' ? '' : (providerInfo?.default_model || '');

    if (setupMode === 'same') {
      setAllAgentsConfig(providerId, defaultModel, false); // Reset web search to false when provider changes
    } else {
      setAgentConfig(activeAgentIndex, providerId, defaultModel, false);
    }
    setModelSearch(''); // Clear model search when provider changes
  };

  // Handle web search toggle
  const handleWebSearchToggle = (enabled: boolean) => {
    if (setupMode === 'same') {
      // Update all agents
      agents.forEach((_, index) => {
        setAgentWebSearch(index, enabled);
      });
    } else {
      setAgentWebSearch(activeAgentIndex, enabled);
    }
  };

  // Handle code execution toggle
  const handleCodeExecutionToggle = (enabled: boolean) => {
    if (setupMode === 'same') {
      // Update all agents
      agents.forEach((_, index) => {
        setAgentCodeExecution(index, enabled);
      });
    } else {
      setAgentCodeExecution(activeAgentIndex, enabled);
    }
  };

  const handleReasoningEffortChange = (value: string) => {
    const normalized = value === '' ? undefined : value as ReasoningEffort;
    if (setupMode === 'same') {
      setAllAgentsReasoningEffort(normalized);
    } else {
      setAgentReasoningEffort(activeAgentIndex, normalized);
    }
  };

  // Handle system message change
  const handleSystemMessageChange = (message: string, agentIndex?: number) => {
    if (setupMode === 'same') {
      if (systemMessageMode === 'same') {
        // Update all agents with same message
        agents.forEach((_, index) => {
          setAgentSystemMessage(index, message);
        });
      } else if (systemMessageMode === 'different' && agentIndex !== undefined) {
        // Update only the specific agent
        setAgentSystemMessage(agentIndex, message);
      }
    } else {
      setAgentSystemMessage(activeAgentIndex, message);
    }
  };

  // Handle system message mode change
  const handleSystemMessageModeChange = (mode: 'skip' | 'same' | 'different') => {
    setSystemMessageMode(mode);
    if (mode === 'skip') {
      // Clear all system messages
      agents.forEach((_, index) => {
        setAgentSystemMessage(index, '');
      });
      setEditingSystemMsgAgent(null);
    } else if (mode === 'same') {
      setEditingSystemMsgAgent(null);
    } else if (mode === 'different') {
      // Start editing first agent
      setEditingSystemMsgAgent(0);
    }
  };

  // Handle model selection
  const handleModelSelect = (model: string) => {
    if (setupMode === 'same') {
      setAllAgentsConfig(currentAgent?.provider || '', model);
    } else {
      setAgentConfig(activeAgentIndex, currentAgent?.provider || '', model);
    }
  };

  if (availableProviders.length === 0) {
    return (
      <motion.div
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        exit={{ opacity: 0, x: -20 }}
        className="space-y-6"
      >
        <div className="bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800 rounded-lg p-6">
          <div className="flex items-center gap-3 mb-3">
            <AlertCircle className="w-6 h-6 text-amber-500" />
            <h3 className="text-lg font-semibold text-amber-700 dark:text-amber-400">
              No API Keys Found
            </h3>
          </div>
          <p className="text-sm text-amber-600 dark:text-amber-400 mb-4">
            Please set up API keys for at least one provider. You can do this by:
          </p>
          <ul className="text-sm text-amber-600 dark:text-amber-400 list-disc list-inside space-y-1">
            <li>Setting environment variables (e.g., OPENAI_API_KEY)</li>
            <li>Running <code className="bg-amber-100 dark:bg-amber-900/40 px-1 rounded">massgen --setup</code> in terminal</li>
            <li>Creating a <code className="bg-amber-100 dark:bg-amber-900/40 px-1 rounded">~/.massgen/.env</code> file</li>
          </ul>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20 }}
      className="space-y-6"
    >
      {/* Header */}
      <div>
        <h2 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-2">
          {setupMode === 'same' ? 'Configure All Agents' : 'Configure Each Agent'}
        </h2>
        <p className="text-sm text-gray-600 dark:text-gray-400">
          {setupMode === 'same'
            ? `All ${agents.length} agents will use the same provider and model.`
            : 'Select a provider and model for each agent.'}
        </p>
      </div>

      {/* Agent tabs for different config mode */}
      {setupMode === 'different' && agents.length > 1 && (
        <div className="flex gap-2 flex-wrap">
          {agents.map((agent, index) => {
            const letter = agent.id.replace('agent_', '').toUpperCase();
            const isConfigured = agent.provider && agent.model;
            return (
              <button
                key={agent.id}
                onClick={() => setActiveAgentIndex(index)}
                className={`px-4 py-2 rounded-lg font-medium text-sm transition-all flex items-center gap-2 ${
                  activeAgentIndex === index
                    ? 'bg-blue-500 text-white'
                    : isConfigured
                    ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 border border-green-300 dark:border-green-700'
                    : 'bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-700'
                }`}
              >
                <Bot className="w-4 h-4" />
                Agent {letter}
                {isConfigured && activeAgentIndex !== index && (
                  <Check className="w-4 h-4" />
                )}
              </button>
            );
          })}
        </div>
      )}

      {/* Current selection summary */}
      {currentAgent?.provider && (
        <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Bot className="w-5 h-5 text-blue-500" />
            <div>
              <span className="text-sm text-gray-500 dark:text-gray-400">Current selection: </span>
              <span className="font-medium text-gray-800 dark:text-gray-200">
                {selectedProvider?.name || currentAgent.provider}
              </span>
              {currentAgent.model && (
                <>
                  <span className="text-gray-400 mx-2">/</span>
                  <span className="font-medium text-green-600 dark:text-green-400">
                    {currentAgent.model}
                  </span>
                </>
              )}
              {currentAgent.enable_web_search && (
                <span className="ml-2 inline-flex items-center gap-1 text-xs px-2 py-0.5 bg-blue-100 dark:bg-blue-900/40 text-blue-700 dark:text-blue-300 rounded-full">
                  <Globe className="w-3 h-3" />
                  Web Search
                </span>
              )}
              {currentAgent.enable_code_execution && (
                <span className="ml-2 inline-flex items-center gap-1 text-xs px-2 py-0.5 bg-purple-100 dark:bg-purple-900/40 text-purple-700 dark:text-purple-300 rounded-full">
                  <Code className="w-3 h-3" />
                  Code Execution
                </span>
              )}
              {currentAgent.system_message && (
                <span className="ml-2 inline-flex items-center gap-1 text-xs px-2 py-0.5 bg-amber-100 dark:bg-amber-900/40 text-amber-700 dark:text-amber-300 rounded-full">
                  <MessageSquare className="w-3 h-3" />
                  Custom Instructions
                </span>
              )}
              {currentAgent.reasoning_effort && (
                <span className="ml-2 inline-flex items-center gap-1 text-xs px-2 py-0.5 bg-indigo-100 dark:bg-indigo-900/40 text-indigo-700 dark:text-indigo-300 rounded-full">
                  <Sparkles className="w-3 h-3" />
                  Reasoning {currentAgent.reasoning_effort}
                </span>
              )}
            </div>
          </div>
          {currentAgent.provider && currentAgent.model && (
            <div className="flex items-center gap-1 text-green-500">
              <Check className="w-5 h-5" />
              <span className="text-sm font-medium">Ready</span>
            </div>
          )}
        </div>
      )}

      {/* Two-column layout: Providers | Models */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Providers Column */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-gray-800 dark:text-gray-200">
              1. Select Provider
            </h3>
            <span className="text-xs text-gray-500">
              {availableProviders.length} available
            </span>
          </div>

          {/* Provider search */}
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
            <input
              type="text"
              value={providerSearch}
              onChange={(e) => setProviderSearch(e.target.value)}
              placeholder="Search providers..."
              className="w-full pl-10 pr-10 py-2.5 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600
                       rounded-lg text-gray-800 dark:text-gray-200 text-sm
                       focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
            />
            {providerSearch && (
              <button
                onClick={() => setProviderSearch('')}
                className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>

          {/* Provider list - scrollable */}
          <div className="space-y-2 max-h-[400px] overflow-y-auto pr-2">
            {filteredAvailableProviders.map((provider) => (
              <ProviderCard
                key={provider.id}
                provider={provider}
                isSelected={currentAgent?.provider === provider.id}
                onSelect={() => handleProviderSelect(provider.id)}
              />
            ))}

            {filteredUnavailableProviders.length > 0 && (
              <>
                <div className="text-xs text-gray-500 uppercase tracking-wide pt-4 pb-2">
                  Unavailable (need API key)
                </div>
                {filteredUnavailableProviders.map((provider) => (
                  <ProviderCard
                    key={provider.id}
                    provider={provider}
                    isSelected={false}
                    onSelect={() => {}}
                    disabled
                  />
                ))}
              </>
            )}

            {filteredAvailableProviders.length === 0 && filteredUnavailableProviders.length === 0 && (
              <div className="text-center py-8 text-gray-500">
                No providers match "{providerSearch}"
              </div>
            )}
          </div>
        </div>

        {/* Models Column */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-gray-800 dark:text-gray-200">
              2. Select Model
            </h3>
            {availableModels.length > 0 && (
              <span className="text-xs text-gray-500">
                {availableModels.length} models
              </span>
            )}
          </div>

          {!currentAgent?.provider ? (
            <div className="flex flex-col items-center justify-center py-12 text-gray-500 bg-gray-50 dark:bg-gray-800/50 rounded-lg border-2 border-dashed border-gray-300 dark:border-gray-700">
              <Bot className="w-12 h-12 mb-3 opacity-50" />
              <p className="text-sm">Select a provider first</p>
            </div>
          ) : loadingModels[currentAgent.provider] ? (
            <div className="flex flex-col items-center justify-center py-12 text-gray-500 bg-gray-50 dark:bg-gray-800/50 rounded-lg">
              <div className="w-8 h-8 border-2 border-blue-500 border-t-transparent rounded-full animate-spin mb-3" />
              <p className="text-sm">Loading models...</p>
            </div>
          ) : (
            <>
              {/* Model search */}
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  type="text"
                  value={modelSearch}
                  onChange={(e) => setModelSearch(e.target.value)}
                  placeholder="Search models or type custom name..."
                  className="w-full pl-10 pr-10 py-2.5 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600
                           rounded-lg text-gray-800 dark:text-gray-200 text-sm
                           focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent"
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && modelSearch.trim()) {
                      handleModelSelect(modelSearch.trim());
                    }
                  }}
                />
                {modelSearch && (
                  <button
                    onClick={() => setModelSearch('')}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600"
                  >
                    <X className="w-4 h-4" />
                  </button>
                )}
              </div>

              {/* Custom model hint */}
              {modelSearch && !filteredModels.includes(modelSearch) && (
                <button
                  onClick={() => handleModelSelect(modelSearch.trim())}
                  className="w-full p-3 rounded-lg border-2 border-dashed border-blue-300 dark:border-blue-700
                           bg-blue-50 dark:bg-blue-900/20 text-left hover:bg-blue-100 dark:hover:bg-blue-900/30 transition-all"
                >
                  <div className="flex items-center gap-2">
                    <span className="text-blue-600 dark:text-blue-400 text-sm">
                      Press Enter or click to use custom model:
                    </span>
                    <span className="font-medium text-blue-700 dark:text-blue-300">
                      {modelSearch}
                    </span>
                  </div>
                </button>
              )}

              {/* Model list - scrollable */}
              <div className="space-y-2 max-h-[350px] overflow-y-auto pr-2">
                {filteredModels.map((model) => (
                  <ModelCard
                    key={model}
                    model={model}
                    isSelected={currentAgent?.model === model}
                    isDefault={model === selectedProvider?.default_model}
                    onSelect={() => handleModelSelect(model)}
                  />
                ))}

                {filteredModels.length === 0 && modelSearch && (
                  <div className="text-center py-4 text-gray-500 text-sm">
                    No models match "{modelSearch}"
                  </div>
                )}
              </div>

              {/* Per-agent options - show after model is selected */}
              {currentAgent?.model && (
                <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700">
                  <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
                    3. Agent Options
                  </h4>
                  {isLoadingCapabilities ? (
                    <div className="flex items-center gap-2 text-gray-400 text-sm p-3">
                      <div className="w-4 h-4 border-2 border-gray-300 border-t-transparent rounded-full animate-spin" />
                      <span>Loading options...</span>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {loadingReasoningProfile && currentAgent.model && (
                        <div className="p-3 bg-gray-50 dark:bg-gray-800/50 rounded-lg">
                          <div className="flex items-center gap-2 text-gray-400 text-sm">
                            <div className="w-4 h-4 border-2 border-gray-300 border-t-transparent rounded-full animate-spin" />
                            <span>Loading reasoning options...</span>
                          </div>
                        </div>
                      )}

                      {reasoningProfile && (
                        <div className="p-3 bg-gray-50 dark:bg-gray-800/50 rounded-lg">
                          <div className="flex items-center gap-2 mb-2">
                            <Sparkles className="w-4 h-4 text-gray-500 dark:text-gray-400" />
                            <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                              Reasoning Effort
                            </span>
                          </div>
                          <select
                            value={currentAgent.reasoning_effort ?? reasoningProfile.default_effort}
                            onChange={(e) => handleReasoningEffortChange(e.target.value)}
                            className="w-full px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-600
                                     rounded-lg text-gray-800 dark:text-gray-200 text-sm
                                     focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                          >
                            {reasoningProfile.choices.map(([label, value]) => (
                              <option key={value} value={value}>
                                {label}
                              </option>
                            ))}
                          </select>
                          <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
                            {reasoningProfile.description}
                          </p>
                        </div>
                      )}

                      {/* Tool toggles - code execution only shown when Docker is NOT enabled */}
                      {(currentCapabilities?.supports_web_search || (!useDocker && currentCapabilities?.supports_code_execution)) && (
                        <div className="space-y-2">
                          {currentCapabilities?.supports_web_search && (
                            <OptionToggle
                              enabled={currentAgent.enable_web_search ?? false}
                              onChange={handleWebSearchToggle}
                              icon={<Globe className="w-4 h-4" />}
                              label="Web Search"
                              description="Search the web for up-to-date information"
                            />
                          )}
                          {/* Only show code execution when Docker is NOT enabled - Docker mode has its own code execution */}
                          {!useDocker && currentCapabilities?.supports_code_execution && (
                            <OptionToggle
                              enabled={currentAgent.enable_code_execution ?? false}
                              onChange={handleCodeExecutionToggle}
                              icon={<Code className="w-4 h-4" />}
                              label="Code Execution"
                              description="Run code in provider's cloud sandbox"
                            />
                          )}
                        </div>
                      )}

                      {/* System message */}
                      <div className="p-3 bg-gray-50 dark:bg-gray-800/50 rounded-lg">
                        <div className="flex items-center gap-2 mb-2">
                          <MessageSquare className="w-4 h-4 text-gray-500 dark:text-gray-400" />
                          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                            System Message
                          </span>
                          <span className="text-xs text-gray-400">(optional)</span>
                        </div>

                        {/* In "same" setup mode with multiple agents, show mode selector */}
                        {setupMode === 'same' && agents.length > 1 ? (
                          <div className="space-y-3">
                            {/* Mode selector */}
                            <div className="flex gap-2 flex-wrap">
                              <button
                                onClick={() => handleSystemMessageModeChange('skip')}
                                className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors ${
                                  systemMessageMode === 'skip'
                                    ? 'bg-blue-500 text-white'
                                    : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
                                }`}
                              >
                                Skip
                              </button>
                              <button
                                onClick={() => handleSystemMessageModeChange('same')}
                                className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors ${
                                  systemMessageMode === 'same'
                                    ? 'bg-blue-500 text-white'
                                    : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
                                }`}
                              >
                                Same for all
                              </button>
                              <button
                                onClick={() => handleSystemMessageModeChange('different')}
                                className={`px-3 py-1.5 text-xs font-medium rounded-lg transition-colors ${
                                  systemMessageMode === 'different'
                                    ? 'bg-blue-500 text-white'
                                    : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
                                }`}
                              >
                                Different per agent
                              </button>
                            </div>

                            {/* Content based on mode */}
                            {systemMessageMode === 'same' && (
                              <textarea
                                value={agents[0]?.system_message ?? ''}
                                onChange={(e) => handleSystemMessageChange(e.target.value)}
                                placeholder="Add custom instructions for all agents..."
                                rows={3}
                                className="w-full px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-600
                                         rounded-lg text-gray-800 dark:text-gray-200 text-sm
                                         focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                                         resize-none placeholder-gray-400 dark:placeholder-gray-500"
                              />
                            )}

                            {systemMessageMode === 'different' && (
                              <div className="space-y-2">
                                {/* Agent tabs */}
                                <div className="flex gap-1 flex-wrap">
                                  {agents.map((agent, index) => {
                                    const letter = agent.id.replace('agent_', '').toUpperCase();
                                    const hasMessage = !!agent.system_message;
                                    return (
                                      <button
                                        key={agent.id}
                                        onClick={() => setEditingSystemMsgAgent(index)}
                                        className={`px-2 py-1 text-xs font-medium rounded transition-colors flex items-center gap-1 ${
                                          editingSystemMsgAgent === index
                                            ? 'bg-blue-500 text-white'
                                            : hasMessage
                                            ? 'bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300'
                                            : 'bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-400'
                                        }`}
                                      >
                                        Agent {letter}
                                        {hasMessage && editingSystemMsgAgent !== index && (
                                          <Check className="w-3 h-3" />
                                        )}
                                      </button>
                                    );
                                  })}
                                </div>

                                {/* Textarea for selected agent */}
                                {editingSystemMsgAgent !== null && (
                                  <textarea
                                    value={agents[editingSystemMsgAgent]?.system_message ?? ''}
                                    onChange={(e) => handleSystemMessageChange(e.target.value, editingSystemMsgAgent)}
                                    placeholder={`Instructions for Agent ${agents[editingSystemMsgAgent]?.id.replace('agent_', '').toUpperCase()}...`}
                                    rows={3}
                                    className="w-full px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-600
                                             rounded-lg text-gray-800 dark:text-gray-200 text-sm
                                             focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                                             resize-none placeholder-gray-400 dark:placeholder-gray-500"
                                  />
                                )}
                              </div>
                            )}

                            {systemMessageMode === 'skip' && (
                              <p className="text-xs text-gray-500 dark:text-gray-400">
                                No custom instructions will be added to agents
                              </p>
                            )}
                          </div>
                        ) : (
                          /* Single agent or "different" setup mode - simple textarea */
                          <>
                            <textarea
                              value={currentAgent.system_message ?? ''}
                              onChange={(e) => handleSystemMessageChange(e.target.value)}
                              placeholder="Add custom instructions for this agent..."
                              rows={3}
                              className="w-full px-3 py-2 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-600
                                       rounded-lg text-gray-800 dark:text-gray-200 text-sm
                                       focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
                                       resize-none placeholder-gray-400 dark:placeholder-gray-500"
                            />
                            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                              Custom instructions to guide the agent's behavior and responses
                            </p>
                          </>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              )}
            </>
          )}
        </div>
      </div>
    </motion.div>
  );
}
