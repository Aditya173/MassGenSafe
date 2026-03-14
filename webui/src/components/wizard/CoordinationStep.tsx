/**
 * Coordination Step Component
 *
 * Keep quickstart coordination aligned with the terminal wizard:
 * mode selection, presenter choice, and decomposition answer caps only.
 */

import { motion } from 'framer-motion';
import { Settings, Vote, Info, ListOrdered, GitBranch } from 'lucide-react';
import { useWizardStore } from '../../stores/wizardStore';

export function CoordinationStep() {
  const coordinationSettings = useWizardStore((s) => s.coordinationSettings);
  const setCoordinationSettings = useWizardStore((s) => s.setCoordinationSettings);
  const agentCount = useWizardStore((s) => s.agentCount);
  const agents = useWizardStore((s) => s.agents);
  const coordinationMode = coordinationSettings.coordination_mode ?? 'voting';
  const defaultPresenterAgent = agents[agents.length - 1]?.id ?? 'agent_a';
  const defaultGlobalCap = Math.max(3, agentCount * 3);

  const handleCoordinationModeChange = (mode: 'voting' | 'decomposition') => {
    if (mode === 'decomposition') {
      setCoordinationSettings({
        coordination_mode: 'decomposition',
        presenter_agent: coordinationSettings.presenter_agent ?? defaultPresenterAgent,
        max_new_answers_per_agent: coordinationSettings.max_new_answers_per_agent ?? 2,
        max_new_answers_global: coordinationSettings.max_new_answers_global ?? defaultGlobalCap,
      });
      return;
    }

    setCoordinationSettings({
      coordination_mode: 'voting',
      presenter_agent: undefined,
      max_new_answers_per_agent: undefined,
      max_new_answers_global: undefined,
    });
  };

  const handleMaxAnswersChange = (value: string) => {
    const num = parseInt(value, 10);
    setCoordinationSettings({
      max_new_answers_per_agent: Number.isNaN(num) || num <= 0 ? undefined : num,
    });
  };

  const handleMaxGlobalAnswersChange = (value: string) => {
    const num = parseInt(value, 10);
    setCoordinationSettings({
      max_new_answers_global: Number.isNaN(num) || num <= 0 ? undefined : num,
    });
  };

  const handlePresenterAgentChange = (value: string) => {
    setCoordinationSettings({ presenter_agent: value || undefined });
  };

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20 }}
      className="space-y-6"
    >
      <div>
        <div className="flex items-center gap-2 mb-2">
          <Settings className="w-5 h-5 text-blue-500" />
          <h2 className="text-xl font-semibold text-gray-800 dark:text-gray-200">
            Coordination Settings
          </h2>
          <span className="text-xs px-2 py-0.5 bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-400 rounded-full">
            Optional
          </span>
        </div>
        <p className="text-sm text-gray-600 dark:text-gray-400">
          Keep quickstart simple: use checklist-gated voting by default, or switch to decomposition
          when you want a presenter and tighter answer caps.
        </p>
      </div>

      <div className="flex items-start gap-3 p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
        <Info className="w-5 h-5 text-blue-500 flex-shrink-0 mt-0.5" />
        <div className="text-sm text-blue-700 dark:text-blue-300">
          Voting uses the built-in checklist defaults automatically. Decomposition adds a presenter,
          a per-agent cap, and a team-wide cap.
        </div>
      </div>

      <div className="p-5 bg-gray-50 dark:bg-gray-800/50 rounded-lg border border-gray-200 dark:border-gray-700">
        <div className="flex items-start gap-3 mb-4">
          <div className="p-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
            {coordinationMode === 'decomposition' ? (
              <GitBranch className="w-5 h-5 text-blue-500" />
            ) : (
              <Vote className="w-5 h-5 text-blue-500" />
            )}
          </div>
          <div className="flex-1">
            <h3 className="font-medium text-gray-800 dark:text-gray-200">Coordination Mode</h3>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-0.5">
              Use voting for independent parallel attempts, or decomposition when agents should split
              the task and hand the final synthesis to a presenter.
            </p>
          </div>
        </div>

        <div className="grid gap-3 md:grid-cols-2">
          <button
            type="button"
            onClick={() => handleCoordinationModeChange('voting')}
            className={`rounded-lg border p-4 text-left transition-all ${
              coordinationMode === 'voting'
                ? 'border-blue-500 bg-blue-50 dark:border-blue-700 dark:bg-blue-900/20'
                : 'border-gray-200 bg-white hover:border-gray-300 dark:border-gray-700 dark:bg-gray-900 dark:hover:border-gray-600'
            }`}
          >
            <div className="flex items-center gap-2 font-medium text-gray-800 dark:text-gray-200">
              <Vote className="w-4 h-4 text-blue-500" />
              Voting
            </div>
            <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
              Each agent works in parallel and the team converges through checklist-gated voting and refinement.
            </p>
          </button>
          <button
            type="button"
            onClick={() => handleCoordinationModeChange('decomposition')}
            className={`rounded-lg border p-4 text-left transition-all ${
              coordinationMode === 'decomposition'
                ? 'border-blue-500 bg-blue-50 dark:border-blue-700 dark:bg-blue-900/20'
                : 'border-gray-200 bg-white hover:border-gray-300 dark:border-gray-700 dark:bg-gray-900 dark:hover:border-gray-600'
            }`}
          >
            <div className="flex items-center gap-2 font-medium text-gray-800 dark:text-gray-200">
              <GitBranch className="w-4 h-4 text-blue-500" />
              Decomposition
            </div>
            <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
              Agents specialize on subtasks, then a presenter agent assembles the final answer.
            </p>
          </button>
        </div>

        {coordinationMode === 'decomposition' && (
          <div className="mt-4 space-y-4 border-t border-gray-200 pt-4 dark:border-gray-700">
            <div className="rounded-lg border border-blue-200 bg-blue-50 px-4 py-3 text-sm text-blue-700 dark:border-blue-800 dark:bg-blue-900/20 dark:text-blue-300">
              Recommended decomposition defaults: presenter = last agent, per-agent cap = 2, global cap = 3 x agents.
            </div>

            <div className="grid gap-4 md:grid-cols-3">
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Presenter Agent
                </label>
                <select
                  value={coordinationSettings.presenter_agent ?? defaultPresenterAgent}
                  onChange={(e) => handlePresenterAgentChange(e.target.value)}
                  className="w-full px-4 py-2.5 bg-white dark:bg-gray-900 border border-gray-300 dark:border-gray-600
                             rounded-lg text-gray-800 dark:text-gray-200 text-sm
                             focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  {agents.map((agent) => (
                    <option key={agent.id} value={agent.id}>
                      {agent.id}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Max Answers per Agent
                </label>
                <div className="relative">
                  <ListOrdered className="pointer-events-none absolute left-3 top-3 h-4 w-4 text-gray-400" />
                  <input
                    type="number"
                    min="1"
                    value={coordinationSettings.max_new_answers_per_agent ?? ''}
                    onChange={(e) => handleMaxAnswersChange(e.target.value)}
                    className="w-full rounded-lg border border-gray-300 bg-white py-2.5 pl-10 pr-4 text-sm text-gray-800
                               placeholder-gray-400 focus:border-transparent focus:outline-none focus:ring-2 focus:ring-blue-500
                               dark:border-gray-600 dark:bg-gray-900 dark:text-gray-200"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Max Answers Globally
                </label>
                <input
                  type="number"
                  min="1"
                  value={coordinationSettings.max_new_answers_global ?? ''}
                  onChange={(e) => handleMaxGlobalAnswersChange(e.target.value)}
                  className="w-full rounded-lg border border-gray-300 bg-white px-4 py-2.5 text-sm text-gray-800
                             placeholder-gray-400 focus:border-transparent focus:outline-none focus:ring-2 focus:ring-blue-500
                             dark:border-gray-600 dark:bg-gray-900 dark:text-gray-200"
                />
              </div>
            </div>
          </div>
        )}
      </div>

      <div className="p-4 bg-gray-100 dark:bg-gray-800 rounded-lg">
        <div className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
          <div>
            <span className="font-medium text-gray-700 dark:text-gray-300">Current settings: </span>
            Coordination is <span className="font-medium text-blue-600 dark:text-blue-400">{coordinationMode}</span>.
            {coordinationMode === 'voting' ? (
              <> Quickstart will use the default checklist-gated voting behavior.</>
            ) : (
              <>
                {' '}Presenter: <span className="font-medium text-blue-600 dark:text-blue-400">
                  {coordinationSettings.presenter_agent ?? defaultPresenterAgent}
                </span>.
                {' '}Per-agent cap: <span className="font-medium text-blue-600 dark:text-blue-400">
                  {coordinationSettings.max_new_answers_per_agent ?? 2}
                </span>.
                {' '}Global cap: <span className="font-medium text-blue-600 dark:text-blue-400">
                  {coordinationSettings.max_new_answers_global ?? defaultGlobalCap}
                </span>.
              </>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  );
}
