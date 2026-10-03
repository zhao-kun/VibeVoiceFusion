"use client";

import Link from "next/link";
import Image from "next/image";
import { usePathname, useRouter } from "next/navigation";
import { useProject } from "@/lib/ProjectContext";
import { useLanguage } from "@/lib/i18n/LanguageContext";
import { useGlobalTask } from "@/lib/GlobalTaskContext";
import { hasActiveTask } from "@/types/task";
import { useState, useEffect, useRef } from "react";

interface MenuItem {
  id: string;
  labelKey: string;
  path: string;
  icon: React.ReactNode;
}

interface MenuGroup {
  id: string;
  labelKey: string;
  items: MenuItem[];
}

const getMenuGroups = (): MenuGroup[] => [
  {
    id: "inference",
    labelKey: "navigation.inference",
    items: [
      {
        id: "speaker-role",
        labelKey: "navigation.speakerRole",
        path: "/speaker-role",
        icon: (
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
          </svg>
        ),
      },
      {
        id: "voice-editor",
        labelKey: "navigation.voiceEditor",
        path: "/voice-editor",
        icon: (
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
          </svg>
        ),
      },
      {
        id: "generate-voice",
        labelKey: "navigation.generateVoice",
        path: "/generate-voice",
        icon: (
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
          </svg>
        ),
      },
      {
        id: "transcription",
        labelKey: "navigation.transcription",
        path: "/transcription",
        icon: (
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
          </svg>
        ),
      },
    ],
  },
  {
    id: "training",
    labelKey: "navigation.training",
    items: [
      {
        id: "dataset",
        labelKey: "navigation.dataset",
        path: "/dataset",
        icon: (
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 7v10c0 2.21 3.582 4 8 4s8-1.79 8-4V7M4 7c0 2.21 3.582 4 8 4s8-1.79 8-4M4 7c0-2.21 3.582-4 8-4s8 1.79 8 4m0 5c0 2.21-3.582 4-8 4s-8-1.79-8-4" />
          </svg>
        ),
      },
      {
        id: "fine-tuning",
        labelKey: "navigation.fineTuning",
        path: "/fine-tuning",
        icon: (
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
          </svg>
        ),
      },
    ],
  },
];

export default function Navigation() {
  const pathname = usePathname();
  const router = useRouter();
  const { currentProject, projects, selectProject } = useProject();
  const { t, locale, setLocale } = useLanguage();
  const { currentTask } = useGlobalTask();
  const [showProjectMenu, setShowProjectMenu] = useState(false);
  const [mounted, setMounted] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const menuGroups = getMenuGroups();

  // Check if there's an active task
  const isTaskRunning = hasActiveTask(currentTask);
  const taskType = currentTask?.type;

  // Only show project-dependent content after client-side mount
  useEffect(() => {
    setMounted(true);
  }, []);

  // Navigate to the appropriate page based on task type
  const handleTaskIconClick = () => {
    if (currentTask) {
      // Quick generation has no project, navigate directly
      if (currentTask.type === 'quick_generation') {
        router.push('/quick-generate');
        return;
      }

      // Standalone transcriptions have no project
      if (currentTask.type === 'transcription' && !currentTask.project_id) {
        router.push('/quick-transcribe');
        return;
      }

      // For project-based tasks, select the project first
      if (currentTask.project_id) {
        if (currentProject?.id !== currentTask.project_id) {
          selectProject(currentTask.project_id);
        }
        // Navigate to the appropriate page based on task type
        if (currentTask.type === 'inference') {
          router.push('/generate-voice');
        } else if (currentTask.type === 'training') {
          router.push('/fine-tuning');
        } else if (currentTask.type === 'transcription') {
          router.push('/transcription');
        }
      }
    }
  };

  // Get the tooltip text based on task type
  const getTaskTooltip = () => {
    if (!isTaskRunning) {
      return t('navigation.noRunningTasks');
    }
    if (taskType === 'inference') {
      return t('navigation.viewRunningInference');
    }
    if (taskType === 'training') {
      return t('navigation.viewRunningTraining');
    }
    if (taskType === 'quick_generation') {
      return t('navigation.viewRunningQuickGeneration');
    }
    if (taskType === 'transcription') {
      return t('navigation.viewRunningTranscription');
    }
    return t('navigation.viewRunningTask');
  };

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setShowProjectMenu(false);
      }
    };

    if (showProjectMenu) {
      document.addEventListener("mousedown", handleClickOutside);
    }

    return () => {
      document.removeEventListener("mousedown", handleClickOutside);
    };
  }, [showProjectMenu]);

  const handleChangeProject = (projectId: string) => {
    selectProject(projectId);
    setShowProjectMenu(false);
  };

  const handleGoHome = () => {
    setShowProjectMenu(false);
    router.push("/");
  };

  return (
    <>
      {/* GitHub Link - Top Right of Page */}
      <a
        href="https://github.com/zhao-kun/vibevoice"
        target="_blank"
        rel="noopener noreferrer"
        className="group fixed top-6 right-6 z-50 p-2 rounded-lg bg-gray-900 hover:bg-gray-800 transition-all duration-200 hover:scale-110 border border-gray-700 shadow-lg"
        title={t('navigation.githubTooltip')}
      >
        <svg className="w-5 h-5 text-white group-hover:text-blue-400 transition-colors" fill="currentColor" viewBox="0 0 24 24">
          <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
        </svg>
      </a>

      <nav className="w-64 bg-gray-900 text-white flex flex-col h-screen fixed left-0 top-0 z-50">
        {/* Logo/Header - Clickable to go home */}
        <div
          className="p-6 border-b border-gray-800 cursor-pointer hover:bg-gray-800/50 transition-colors"
          onClick={() => router.push('/')}
          title={t('navigation.goToHome')}
        >
          <div className="flex items-center gap-3 mb-3">
            {/* Logo */}
            <Image
              src="/icon-rect-pulse.svg"
              alt="VibeVoice Logo"
              width={40}
              height={40}
              className="w-10 h-10 flex-shrink-0"
            />
            <div>
              <h1 className="text-xl font-bold text-white">{t('app.title')}</h1>
              <p className="text-xs text-gray-400 mt-1">{t('app.subtitle')}</p>
            </div>
          </div>
        </div>

      {/* Current Project Display */}
      <div className="px-4 py-3 border-b border-gray-800 bg-gray-800/50 relative z-50">
        <div className="text-xs text-gray-400 mb-1">{t('navigation.currentProject')}</div>
        <div className="relative" ref={dropdownRef}>
          <button
            onClick={() => setShowProjectMenu(!showProjectMenu)}
            className="w-full flex items-center justify-between px-3 py-2 bg-gray-700 hover:bg-gray-600 rounded-lg transition-colors text-left"
          >
            <div className="flex items-center space-x-2 flex-1 min-w-0">
              <div className="w-6 h-6 bg-blue-500 rounded flex items-center justify-center text-xs font-bold flex-shrink-0">
                {mounted && currentProject ? currentProject.name.charAt(0).toUpperCase() : '?'}
              </div>
              <span className="text-sm font-medium text-white truncate">
                {mounted && currentProject ? currentProject.name : t('common.loading')}
              </span>
            </div>
            <svg className="w-4 h-4 text-gray-400 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
            </svg>
          </button>

          {/* Project Dropdown */}
          {showProjectMenu && (
            <div className="absolute top-full left-0 right-0 mt-2 bg-gray-800 rounded-lg shadow-xl border border-gray-700 overflow-hidden z-[100]">
              <div className="max-h-64 overflow-y-auto">
                {projects.map((project) => (
                  <button
                    key={project.id}
                    onClick={() => handleChangeProject(project.id)}
                    className={`w-full flex items-center space-x-2 px-3 py-2 hover:bg-gray-700 transition-colors text-left ${
                      currentProject?.id === project.id ? "bg-gray-700" : ""
                    }`}
                  >
                    <div className="w-6 h-6 bg-blue-500 rounded flex items-center justify-center text-xs font-bold flex-shrink-0">
                      {project.name.charAt(0).toUpperCase()}
                    </div>
                    <span className="text-sm text-white truncate">{project.name}</span>
                    {currentProject?.id === project.id && (
                      <svg className="w-4 h-4 text-blue-400 ml-auto flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                      </svg>
                    )}
                  </button>
                ))}
              </div>
              <div className="border-t border-gray-700">
                <button
                  onClick={handleGoHome}
                  className="w-full px-3 py-2 hover:bg-gray-700 transition-colors text-left flex items-center space-x-2 text-blue-400"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                  </svg>
                  <span className="text-sm font-medium">{t('navigation.newProject')}</span>
                </button>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Menu Items */}
      <div className="flex-1 py-4 overflow-y-auto">
        {menuGroups.map((group, groupIndex) => (
          <div key={group.id} className={groupIndex > 0 ? "mt-6" : ""}>
            {/* Group Header */}
            <div className="px-6 mb-2">
              <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
                {t(group.labelKey)}
              </h3>
            </div>

            {/* Group Items */}
            {group.items.map((item) => {
              const isActive = pathname === item.path;

              return (
                <Link
                  key={item.id}
                  href={item.path}
                  className={`
                    flex items-center space-x-3 px-6 py-3 transition-all duration-200
                    relative
                    ${
                      isActive
                        ? "bg-blue-600 text-white"
                        : "text-gray-300 hover:bg-gray-800 hover:text-white"
                    }
                  `}
                >
                  {/* Active indicator */}
                  {isActive && (
                    <div className="absolute left-0 top-0 bottom-0 w-1 bg-blue-400" />
                  )}

                  <div className={isActive ? "text-white" : "text-gray-400"}>
                    {item.icon}
                  </div>
                  <span className="font-medium text-sm">{t(item.labelKey)}</span>
                </Link>
              );
            })}
          </div>
        ))}
      </div>

      {/* Footer */}
      <div className="p-6 border-t border-gray-800 space-y-4">
        {/* Language Switcher */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setLocale('en')}
            className={`flex-1 px-3 py-1.5 text-xs rounded-lg transition-all ${
              locale === 'en'
                ? 'bg-blue-600 text-white font-medium'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-gray-300'
            }`}
          >
            {t('language.en')}
          </button>
          <button
            onClick={() => setLocale('zh')}
            className={`flex-1 px-3 py-1.5 text-xs rounded-lg transition-all ${
              locale === 'zh'
                ? 'bg-blue-600 text-white font-medium'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700 hover:text-gray-300'
            }`}
          >
            {t('language.zh')}
          </button>
        </div>

        <div className="flex items-center justify-between gap-3">
          {/* Version Info */}
          <div className="text-xs text-gray-500 flex-1">
            <p>{process.env.NEXT_PUBLIC_APP_VERSION || 'dev'}</p>
            <p className="mt-1">{t('app.copyright')}</p>
          </div>

          {/* Task Status Icon */}
          {isTaskRunning && (
            <button
              onClick={handleTaskIconClick}
              className={`relative p-2 rounded-lg transition-all cursor-pointer ${
                taskType === 'inference'
                  ? 'bg-blue-600 hover:bg-blue-700 text-white'
                  : taskType === 'quick_generation'
                  ? 'bg-green-600 hover:bg-green-700 text-white'
                  : taskType === 'transcription'
                  ? 'bg-amber-600 hover:bg-amber-700 text-white'
                  : 'bg-purple-600 hover:bg-purple-700 text-white'
              }`}
              title={getTaskTooltip()}
            >
              {/* Icon based on task type */}
              {taskType === 'inference' ? (
                // Microphone/Generation Icon
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                </svg>
              ) : taskType === 'quick_generation' ? (
                // Lightning/Quick Generation Icon
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              ) : taskType === 'transcription' ? (
                // Document/Transcription Icon
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              ) : (
                // Training/Learning Icon
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                </svg>
              )}
              {/* Animated pulse indicator */}
              <span className="absolute -top-1 -right-1 flex h-3 w-3">
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-red-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-3 w-3 bg-red-500"></span>
              </span>
            </button>
          )}
        </div>
      </div>
    </nav>
    </>
  );
}
