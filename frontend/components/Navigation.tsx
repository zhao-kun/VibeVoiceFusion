"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useProject } from "@/lib/ProjectContext";
import { useLanguage } from "@/lib/i18n/LanguageContext";
import { useGlobalTask } from "@/lib/GlobalTaskContext";
import { useState, useEffect, useRef } from "react";

interface MenuItem {
  id: string;
  labelKey: string;
  path: string;
  icon: React.ReactNode;
}

const getMenuItems = (): MenuItem[] => [
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
];

export default function Navigation() {
  const pathname = usePathname();
  const router = useRouter();
  const { currentProject, projects, selectProject } = useProject();
  const { t, locale, setLocale } = useLanguage();
  const { currentTask: runningGeneration } = useGlobalTask();
  const [showProjectMenu, setShowProjectMenu] = useState(false);
  const [mounted, setMounted] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const menuItems = getMenuItems();

  // Only show project-dependent content after client-side mount
  useEffect(() => {
    setMounted(true);
  }, []);

  // Navigate to generation page of project with running task
  const handleTaskIconClick = () => {
    if (runningGeneration && runningGeneration.project_id) {
      // Select the project if it's different
      if (currentProject?.id !== runningGeneration.project_id) {
        selectProject(runningGeneration.project_id);
      }
      // Navigate to generate-voice page
      router.push('/generate-voice');
    }
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
    <nav className="w-64 bg-gray-900 text-white flex flex-col h-screen fixed left-0 top-0 z-50">
      {/* Logo/Header */}
      <div className="p-6 border-b border-gray-800 relative">
        {/* GitHub Link - Top Right */}
        <a
          href="https://github.com/zhao-kun/vibevoice"
          target="_blank"
          rel="noopener noreferrer"
          className="group absolute top-4 right-4 p-2 rounded-lg bg-white/10 hover:bg-white/20 transition-all duration-200 hover:scale-110 border border-white/20"
          title={t('navigation.githubTooltip')}
        >
          <svg className="w-5 h-5 text-white group-hover:text-blue-300 transition-colors" fill="currentColor" viewBox="0 0 24 24">
            <path fillRule="evenodd" d="M12 2C6.477 2 2 6.484 2 12.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0112 6.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.202 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.943.359.309.678.92.678 1.855 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0022 12.017C22 6.484 17.522 2 12 2z" clipRule="evenodd" />
          </svg>
        </a>

        <div className="flex items-center gap-3 mb-3">
          {/* Logo */}
          <img
            src="/icon-rect-pulse.svg"
            alt="VibeVoice Logo"
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
      <div className="flex-1 py-4">
        {menuItems.map((item) => {
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
            <p>{t('app.version')}</p>
            <p className="mt-1">{t('app.copyright')}</p>
          </div>

          {/* Task Status Icon */}
          <button
            onClick={handleTaskIconClick}
            disabled={!runningGeneration}
            className={`relative p-2 rounded-lg transition-all ${
              runningGeneration
                ? 'bg-blue-600 hover:bg-blue-700 text-white cursor-pointer'
                : 'bg-gray-800 text-gray-600 cursor-not-allowed'
            }`}
            title={runningGeneration ? t('navigation.viewRunningTask') : t('navigation.noRunningTasks')}
          >
            {/* Task/Activity Icon */}
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            {/* Badge */}
            {runningGeneration && (
              <span className="absolute -top-1 -right-1 w-4 h-4 bg-red-500 text-white text-[10px] font-bold rounded-full flex items-center justify-center">
                1
              </span>
            )}
          </button>
        </div>
      </div>
    </nav>
  );
}
