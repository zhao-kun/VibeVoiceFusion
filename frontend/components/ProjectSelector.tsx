"use client";

import { useState, useMemo } from "react";
import Image from "next/image";
import { useProject } from "@/lib/ProjectContext";
import { useLanguage } from "@/lib/i18n/LanguageContext";
import { useRouter } from "next/navigation";
import toast from "react-hot-toast";
import PresetVoiceManager from "@/components/PresetVoiceManager";

/**
 * Validate project name according to backend rules:
 * - Must start with an alphabet character (a-z, A-Z)
 * - Can include: alphabet, numbers, underscore (_), hyphen (-), and space
 * - Spaces can only appear in the middle (not at start or end)
 * - Pattern: ^[a-zA-Z][a-zA-Z0-9_\- ]*$
 */
function validateProjectName(name: string, t: (key: string) => string): string | null {
  // Don't show error for empty input (let the required check handle it)
  if (!name) return null;

  // Check if name starts or ends with space
  if (name.startsWith(' ') || name.endsWith(' ')) {
    return t('project.nameNoSpacesAtEnds');
  }

  // Check if first character is an alphabet
  if (!/^[a-zA-Z]/.test(name)) {
    return t('project.nameMustStartWithLetter');
  }

  // Check if all characters are valid (alphabet, number, _, -, or space)
  const validPattern = /^[a-zA-Z][a-zA-Z0-9_\- ]*$/;
  if (!validPattern.test(name)) {
    return t('project.invalidCharsError');
  }

  return null;
}

export default function ProjectSelector() {
  const { projects, selectProject, createProject, deleteProject } = useProject();
  const { t, locale, setLocale } = useLanguage();
  const router = useRouter();
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [newProjectName, setNewProjectName] = useState("");
  const [newProjectDescription, setNewProjectDescription] = useState("");
  const [showPresetManager, setShowPresetManager] = useState(false);

  // Validate project name and return error message if invalid
  const projectNameError = useMemo(() => {
    return validateProjectName(newProjectName, t);
  }, [newProjectName, t]);

  const handleSelectProject = (projectId: string) => {
    selectProject(projectId);
    router.push("/speaker-role");
  };

  const handleCreateProject = async () => {
    if (newProjectName.trim() && !projectNameError) {
      try {
        await createProject(newProjectName.trim(), newProjectDescription.trim());
        setNewProjectName("");
        setNewProjectDescription("");
        setShowCreateModal(false);
        // Will automatically navigate via the context update
        setTimeout(() => router.push("/speaker-role"), 100);
      } catch {
        toast.error(t('project.createError'));
      }
    }
  };

  const handleDeleteProject = async (projectId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    if (confirm(t('project.deleteConfirm'))) {
      try {
        await deleteProject(projectId);
      } catch {
        toast.error(t('project.deleteError'));
      }
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50 flex items-center justify-center p-8 relative">
      {/* Top Right Controls */}
      <div className="absolute top-6 right-6 flex items-center gap-3">
        {/* Preset Voice Manager Button */}
        <button
          onClick={() => setShowPresetManager(true)}
          className="flex items-center gap-2 px-4 py-2 bg-white rounded-lg shadow-md hover:shadow-lg transition-shadow text-gray-700 hover:text-blue-600"
          title={t('presetVoice.title')}
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
          </svg>
          <span className="text-sm font-medium">{t('presetVoice.manageButton')}</span>
        </button>

        {/* Language Switcher */}
        <div className="flex items-center gap-2 bg-white rounded-lg shadow-md p-1">
          <button
            onClick={() => setLocale('en')}
            className={`px-4 py-2 text-sm font-medium rounded-md transition-all ${
              locale === 'en'
                ? 'bg-blue-600 text-white'
                : 'text-gray-600 hover:bg-gray-100'
            }`}
          >
            {t('language.en')}
          </button>
          <button
            onClick={() => setLocale('zh')}
            className={`px-4 py-2 text-sm font-medium rounded-md transition-all ${
              locale === 'zh'
                ? 'bg-blue-600 text-white'
                : 'text-gray-600 hover:bg-gray-100'
            }`}
          >
            {t('language.zh')}
          </button>
        </div>
      </div>

      <div className="max-w-6xl w-full">
        {/* Header */}
        <div className="text-center mb-12">
          {/* Logo */}
          <div className="flex justify-center mb-6">
            <Image
              src="/icon-rect-pulse.svg"
              alt="VibeVoice Logo"
              width={96}
              height={96}
              className="w-24 h-24"
            />
          </div>
          <h1 className="text-5xl font-bold text-gray-900 mb-4">{t('app.title')}</h1>
          <p className="text-xl text-gray-600">{t('app.subtitle')}</p>
          <p className="text-sm text-gray-500 mt-2">{t('project.selectProject')}</p>
        </div>

        {/* Projects Section Header */}
        <div className="flex items-center gap-2 mb-4">
          <h2 className="text-lg font-semibold text-gray-700">{t('project.yourProjects')}</h2>
          <span className="text-sm text-gray-500">({projects.length})</span>
        </div>

        {/* Projects Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
          {projects.map((project) => (
            <div
              key={project.id}
              onClick={() => handleSelectProject(project.id)}
              className="bg-white rounded-xl shadow-md hover:shadow-xl transition-all duration-200 p-6 cursor-pointer border-2 border-transparent hover:border-blue-500 group"
            >
              <div className="flex items-start justify-between mb-4">
                <div className="w-12 h-12 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center text-white font-bold text-xl">
                  {project.name.charAt(0).toUpperCase()}
                </div>
                {projects.length > 1 && (
                  <button
                    onClick={(e) => handleDeleteProject(project.id, e)}
                    className="opacity-0 group-hover:opacity-100 transition-opacity p-2 hover:bg-red-50 rounded-lg text-red-600"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                  </button>
                )}
              </div>

              <h3 className="text-xl font-semibold text-gray-900 mb-2">{project.name}</h3>
              <p className="text-sm text-gray-600 mb-4 line-clamp-2">
                {project.description || t('project.noDescription')}
              </p>

              <div className="flex items-center text-xs text-gray-500 space-x-4">
                <div className="flex items-center space-x-1">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span>{t('project.updated')} {new Date(project.updatedAt).toLocaleDateString()}</span>
                </div>
              </div>
            </div>
          ))}

          {/* Create New Project Card */}
          <div
            onClick={() => setShowCreateModal(true)}
            className="bg-gradient-to-br from-blue-500 to-purple-600 rounded-xl shadow-md hover:shadow-xl transition-all duration-200 p-6 cursor-pointer flex flex-col items-center justify-center text-white min-h-[200px] group hover:scale-105"
          >
            <div className="w-16 h-16 bg-white/20 rounded-full flex items-center justify-center mb-4 group-hover:bg-white/30 transition-colors">
              <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
              </svg>
            </div>
            <h3 className="text-xl font-semibold mb-2">{t('project.createNew')}</h3>
            <p className="text-sm text-white/80">{t('project.selectProject')}</p>
          </div>

          {/* Quick Generate Card */}
          <div
            onClick={() => router.push('/quick-generate')}
            className="bg-gradient-to-br from-emerald-500 to-teal-600 rounded-xl shadow-md hover:shadow-xl transition-all duration-200 p-6 cursor-pointer flex flex-col items-center justify-center text-white min-h-[200px] group hover:scale-105"
          >
            <div className="w-16 h-16 bg-white/20 rounded-full flex items-center justify-center mb-4 group-hover:bg-white/30 transition-colors">
              <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
              </svg>
            </div>
            <h3 className="text-xl font-semibold mb-2">{t('quickGenerate.cardTitle')}</h3>
            <p className="text-sm text-white/80">{t('quickGenerate.cardDescription')}</p>
          </div>

          {/* Quick Transcribe Card */}
          <div
            onClick={() => router.push('/quick-transcribe')}
            className="bg-gradient-to-br from-amber-500 to-orange-600 rounded-xl shadow-md hover:shadow-xl transition-all duration-200 p-6 cursor-pointer flex flex-col items-center justify-center text-white min-h-[200px] group hover:scale-105"
          >
            <div className="w-16 h-16 bg-white/20 rounded-full flex items-center justify-center mb-4 group-hover:bg-white/30 transition-colors">
              <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
            </div>
            <h3 className="text-xl font-semibold mb-2">{t('transcription.cardTitle')}</h3>
            <p className="text-sm text-white/80">{t('transcription.cardDescription')}</p>
          </div>
        </div>

        {/* Create Project Modal */}
        {showCreateModal && (
          <div className="fixed inset-0 bg-gray-900/50 flex items-center justify-center z-50 p-4">
            <div className="bg-white rounded-xl shadow-xl max-w-md w-full p-6">
              <h2 className="text-2xl font-bold text-gray-900 mb-4">{t('project.createNew')}</h2>

              <div className="space-y-4 mb-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    {t('project.projectName')} *
                  </label>
                  <input
                    type="text"
                    value={newProjectName}
                    onChange={(e) => setNewProjectName(e.target.value)}
                    placeholder={t('project.enterProjectName')}
                    className={`w-full px-4 py-2 border rounded-lg focus:outline-none focus:ring-2 ${
                      projectNameError
                        ? 'border-red-300 focus:ring-red-500'
                        : 'border-gray-300 focus:ring-blue-500'
                    }`}
                    autoFocus
                  />
                  {projectNameError ? (
                    <p className="text-xs text-red-600 mt-1">{projectNameError}</p>
                  ) : (
                    <p className="text-xs text-gray-500 mt-1">{t('project.validNameHint')}</p>
                  )}
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    {t('project.projectDescription')}
                  </label>
                  <textarea
                    value={newProjectDescription}
                    onChange={(e) => setNewProjectDescription(e.target.value)}
                    placeholder={t('project.enterProjectDescription')}
                    className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
                    rows={3}
                  />
                </div>
              </div>

              <div className="flex space-x-3">
                <button
                  onClick={() => {
                    setShowCreateModal(false);
                    setNewProjectName("");
                    setNewProjectDescription("");
                  }}
                  className="flex-1 px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors font-medium"
                >
                  {t('common.cancel')}
                </button>
                <button
                  onClick={handleCreateProject}
                  disabled={!newProjectName.trim() || !!projectNameError}
                  className="flex-1 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {t('project.createProject')}
                </button>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Version Info - Bottom Right */}
      <div className="absolute bottom-6 right-6 text-xs text-gray-400">
        {process.env.NEXT_PUBLIC_APP_VERSION || 'dev'}
      </div>

      {/* Preset Voice Manager Modal */}
      <PresetVoiceManager
        isOpen={showPresetManager}
        onClose={() => setShowPresetManager(false)}
      />
    </div>
  );
}
