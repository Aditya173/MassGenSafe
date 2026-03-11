/**
 * API Key Step Component
 *
 * Shows which providers have API keys configured and provides setup instructions.
 */

import { motion } from 'framer-motion';
import { Key, Check, AlertTriangle, RefreshCw, ExternalLink, Bot } from 'lucide-react';
import { useWizardStore } from '../../stores/wizardStore';

const AGENT_FRAMEWORK_PROVIDER_IDS = new Set(['claude_code', 'codex', 'copilot']);

export function ApiKeyStep() {
  const providers = useWizardStore((s) => s.providers);
  const isLoading = useWizardStore((s) => s.isLoading);
  const fetchProviders = useWizardStore((s) => s.fetchProviders);

  const availableProviders = providers.filter((p) => p.has_api_key);
  const unavailableProviders = providers.filter((p) => !p.has_api_key);

  const isAgentFrameworkProvider = (providerId: string, isAgentFramework?: boolean) =>
    Boolean(isAgentFramework) || AGENT_FRAMEWORK_PROVIDER_IDS.has(providerId);

  const handleRefresh = () => {
    fetchProviders();
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20 }}
      className="space-y-6"
    >
      <div>
        <h2 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-2">
          API Keys
        </h2>
        <p className="text-sm text-gray-600 dark:text-gray-400">
          MassGen needs API keys to access LLM providers. Set up at least one provider to continue.
        </p>
      </div>

      {/* Available Providers */}
      {availableProviders.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Available Providers
          </h3>
          <div className="space-y-2">
            {availableProviders.map((provider) => (
              <div
                key={provider.id}
                className="flex items-center gap-3 p-3 bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg"
              >
                <Check className="w-5 h-5 text-green-600 dark:text-green-400" />
                <div className="flex items-center gap-2">
                  <span className="text-gray-800 dark:text-gray-200 font-medium">
                    {provider.name}
                  </span>
                  {isAgentFrameworkProvider(provider.id, provider.is_agent_framework) && (
                    <span className="inline-flex items-center gap-1 rounded-full bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-800 dark:bg-amber-900/40 dark:text-amber-200">
                      <Bot className="h-3 w-3" />
                      Agent
                    </span>
                  )}
                </div>
                <span className="text-xs text-green-600 dark:text-green-400">
                  API key configured
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Unavailable Providers */}
      {unavailableProviders.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Not Configured
          </h3>
          <div className="space-y-2">
            {unavailableProviders.map((provider) => (
              <div
                key={provider.id}
                className="flex items-center gap-3 p-3 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg"
              >
                <AlertTriangle className="w-5 h-5 text-amber-500" />
                <div className="flex items-center gap-2">
                  <span className="text-gray-600 dark:text-gray-400">
                    {provider.name}
                  </span>
                  {isAgentFrameworkProvider(provider.id, provider.is_agent_framework) && (
                    <span className="inline-flex items-center gap-1 rounded-full bg-amber-100 px-2 py-0.5 text-xs font-medium text-amber-800 dark:bg-amber-900/40 dark:text-amber-200">
                      <Bot className="h-3 w-3" />
                      Agent
                    </span>
                  )}
                </div>
                {provider.env_var && (
                  <code className="text-xs bg-gray-200 dark:bg-gray-700 px-2 py-0.5 rounded">
                    {provider.env_var}
                  </code>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Setup Instructions */}
      <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4 space-y-3">
        <div className="flex items-center gap-2">
          <Key className="w-5 h-5 text-blue-600 dark:text-blue-400" />
          <h3 className="text-sm font-medium text-blue-800 dark:text-blue-200">
            How to Set API Keys
          </h3>
        </div>
        <div className="text-sm text-blue-700 dark:text-blue-300 space-y-2">
          <p>In your terminal, set environment variables before running MassGen:</p>
          <div className="bg-blue-100 dark:bg-blue-900/40 rounded p-3 font-mono text-xs overflow-x-auto">
            <div># OpenAI</div>
            <div>export OPENAI_API_KEY=sk-your-key-here</div>
            <div className="mt-2"># Anthropic (Claude)</div>
            <div>export ANTHROPIC_API_KEY=sk-ant-your-key-here</div>
            <div className="mt-2"># Google (Gemini)</div>
            <div>export GOOGLE_API_KEY=your-key-here</div>
          </div>
          <p className="text-xs">
            Then restart the MassGen server for changes to take effect.
          </p>
        </div>
        <a
          href="https://massgen.readthedocs.io/en/latest/quickstart/setup.html"
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1 text-sm text-blue-600 dark:text-blue-400 hover:underline"
        >
          <ExternalLink className="w-4 h-4" />
          View full setup guide
        </a>
      </div>

      {/* Refresh Button */}
      <div className="flex items-center justify-between">
        <button
          onClick={handleRefresh}
          disabled={isLoading}
          className="flex items-center gap-2 px-4 py-2 text-sm text-gray-600 dark:text-gray-400
                   hover:text-gray-800 dark:hover:text-gray-200 transition-colors
                   disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
          <span>Refresh</span>
        </button>
        {availableProviders.length === 0 && (
          <span className="text-sm text-amber-600 dark:text-amber-400">
            Set at least one API key to continue
          </span>
        )}
      </div>
    </motion.div>
  );
}
