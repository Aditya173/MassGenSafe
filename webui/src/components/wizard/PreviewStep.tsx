/**
 * Preview Step Component
 *
 * Final step - preview generated config and save.
 * Allows editing both the filename and the YAML content.
 */

import { useState } from 'react';
import { motion } from 'framer-motion';
import { FileText, Check, AlertCircle, Loader2, Pencil, Code } from 'lucide-react';
import { useWizardStore } from '../../stores/wizardStore';

export function PreviewStep() {
  const generatedYaml = useWizardStore((s) => s.generatedYaml);
  const isLoading = useWizardStore((s) => s.isLoading);
  const error = useWizardStore((s) => s.error);
  const setupStatus = useWizardStore((s) => s.setupStatus);
  const configFilename = useWizardStore((s) => s.configFilename);
  const configSaveLocation = useWizardStore((s) => s.configSaveLocation);
  const setConfigFilename = useWizardStore((s) => s.setConfigFilename);
  const setConfigSaveLocation = useWizardStore((s) => s.setConfigSaveLocation);
  const setGeneratedYaml = useWizardStore((s) => s.setGeneratedYaml);

  // Toggle between view and edit mode
  const [isEditing, setIsEditing] = useState(false);
  const [editedYaml, setEditedYaml] = useState(generatedYaml || '');

  // Sync editedYaml when generatedYaml changes
  if (generatedYaml && editedYaml === '' && generatedYaml !== editedYaml) {
    setEditedYaml(generatedYaml);
  }

  const handleSaveEdit = () => {
    setGeneratedYaml(editedYaml);
    setIsEditing(false);
  };

  const handleCancelEdit = () => {
    setEditedYaml(generatedYaml || '');
    setIsEditing(false);
  };

  if (isLoading) {
    return (
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="flex flex-col items-center justify-center py-12"
      >
        <Loader2 className="w-12 h-12 text-blue-500 animate-spin mb-4" />
        <p className="text-gray-600 dark:text-gray-400">Generating configuration...</p>
      </motion.div>
    );
  }

  if (error) {
    return (
      <motion.div
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        className="space-y-6"
      >
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-6">
          <div className="flex items-center gap-3 mb-3">
            <AlertCircle className="w-6 h-6 text-red-500" />
            <h3 className="text-lg font-semibold text-red-700 dark:text-red-400">
              Error Generating Config
            </h3>
          </div>
          <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
        </div>
      </motion.div>
    );
  }

  const projectConfigDir = setupStatus?.config_path?.includes('/.massgen/')
    ? setupStatus.config_path.replace(/\/[^/]+$/, '')
    : '.massgen';
  const configDir = configSaveLocation === 'project' ? projectConfigDir : '~/.config/massgen';

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      exit={{ opacity: 0, x: -20 }}
      className="space-y-6"
    >
      <div>
        <h2 className="text-xl font-semibold text-gray-800 dark:text-gray-200 mb-2">
          Review Configuration
        </h2>
        <p className="text-sm text-gray-600 dark:text-gray-400">
          Review and customize your configuration before saving.
        </p>
      </div>

      {/* Config Filename Input */}
      <div className="bg-gray-50 dark:bg-gray-800/50 rounded-lg p-4">
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Save Location
          </label>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <button
              type="button"
              onClick={() => setConfigSaveLocation('project')}
              className={`p-3 rounded-lg border text-left transition-colors ${
                configSaveLocation === 'project'
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300'
                  : 'border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-700 dark:text-gray-300'
              }`}
            >
              <div className="font-medium text-sm">Project</div>
              <div className="text-xs text-gray-500 dark:text-gray-400">Save to <code>.massgen/</code> in this workspace</div>
            </button>
            <button
              type="button"
              onClick={() => setConfigSaveLocation('global')}
              className={`p-3 rounded-lg border text-left transition-colors ${
                configSaveLocation === 'global'
                  ? 'border-blue-500 bg-blue-50 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300'
                  : 'border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-700 dark:text-gray-300'
              }`}
            >
              <div className="font-medium text-sm">Global</div>
              <div className="text-xs text-gray-500 dark:text-gray-400">Save to <code>~/.config/massgen/</code></div>
            </button>
          </div>
        </div>
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          <Pencil className="w-4 h-4 inline mr-1" />
          Config Name
        </label>
        <div className="flex items-center gap-2">
          <input
            type="text"
            value={configFilename}
            onChange={(e) => setConfigFilename(e.target.value)}
            placeholder="config"
            className="flex-1 px-3 py-2 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600
                     rounded-lg text-gray-800 dark:text-gray-200 text-sm
                     focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
          <span className="text-gray-500 dark:text-gray-400 text-sm">.yaml</span>
        </div>
        <p className="text-xs text-gray-500 mt-2">
          Will be saved to:{' '}
          <code className="bg-gray-100 dark:bg-gray-700 px-1 rounded">
            {configDir}/{configFilename || 'config'}.yaml
          </code>
        </p>
      </div>

      {/* YAML Content */}
      {generatedYaml && (
        <div className="bg-gray-900 rounded-lg overflow-hidden">
          <div className="flex items-center justify-between px-4 py-2 bg-gray-800 border-b border-gray-700">
            <div className="flex items-center gap-2">
              <FileText className="w-4 h-4 text-gray-400" />
              <span className="text-sm text-gray-300">{configFilename || 'config'}.yaml</span>
            </div>
            <div className="flex items-center gap-2">
              {isEditing ? (
                <>
                  <button
                    onClick={handleCancelEdit}
                    className="px-3 py-1 text-xs text-gray-400 hover:text-gray-200 transition-colors"
                  >
                    Cancel
                  </button>
                  <button
                    onClick={handleSaveEdit}
                    className="px-3 py-1 text-xs bg-green-600 hover:bg-green-500 text-white rounded transition-colors"
                  >
                    Apply Changes
                  </button>
                </>
              ) : (
                <button
                  onClick={() => setIsEditing(true)}
                  className="flex items-center gap-1 px-3 py-1 text-xs text-gray-400 hover:text-gray-200
                           bg-gray-700 hover:bg-gray-600 rounded transition-colors"
                >
                  <Code className="w-3 h-3" />
                  Edit YAML
                </button>
              )}
            </div>
          </div>

          {isEditing ? (
            <textarea
              value={editedYaml}
              onChange={(e) => setEditedYaml(e.target.value)}
              className="w-full h-[350px] p-4 bg-gray-900 text-gray-300 font-mono text-sm
                       resize-none focus:outline-none border-none"
              spellCheck={false}
            />
          ) : (
            <pre className="p-4 text-sm text-gray-300 overflow-x-auto max-h-[350px] overflow-y-auto">
              <code>{generatedYaml}</code>
            </pre>
          )}
        </div>
      )}

      <div className="bg-green-50 dark:bg-green-900/20 border border-green-200 dark:border-green-800 rounded-lg p-4">
        <div className="flex items-center gap-2">
          <Check className="w-5 h-5 text-green-500" />
          <span className="text-sm text-green-700 dark:text-green-400">
            Click "Save & Start" to save this configuration and begin using MassGen.
          </span>
        </div>
      </div>
    </motion.div>
  );
}
