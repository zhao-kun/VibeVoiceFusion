"use client";

import Link from "next/link";
import Image from "next/image";
import { usePathname, useRouter } from "next/navigation";
import { useLanguage } from "@/lib/i18n/LanguageContext";
import { useGlobalTask } from "@/lib/GlobalTaskContext";
import { hasActiveTask } from "@/types/task";

export default function QuickGenerateNavigation() {
  const router = useRouter();
  const pathname = usePathname();
  const { t, locale, setLocale } = useLanguage();
  const { currentTask } = useGlobalTask();

  // Check if there's an active task
  const isTaskRunning = hasActiveTask(currentTask);
  const taskType = currentTask?.type;

  // Navigate to the appropriate page based on task type
  const handleTaskIconClick = () => {
    if (currentTask) {
      if (currentTask.type === 'quick_generation') {
        router.push('/quick-generate');
        return;
      }

      if (currentTask.type === 'transcription' && !currentTask.project_id) {
        router.push('/quick-transcribe');
        return;
      }

      // For project-based tasks, navigate to appropriate page
      if (currentTask.type === 'inference') {
        router.push('/generate-voice');
      } else if (currentTask.type === 'training') {
        router.push('/fine-tuning');
      } else if (currentTask.type === 'transcription') {
        router.push('/transcription');
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

        {/* Quick Generate Mode Indicator */}
        <div className="px-4 py-3 border-b border-gray-800 bg-gradient-to-r from-emerald-900/50 to-teal-900/50">
          <div className="flex items-center gap-2">
            <svg className="w-5 h-5 text-emerald-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
            <span className="text-sm font-medium text-emerald-300">
              {pathname === '/quick-transcribe' ? t('transcription.modeTitle') : t('quickGenerate.modeTitle')}
            </span>
          </div>
          <p className="text-xs text-gray-400 mt-1">{t('quickGenerate.modeDescription')}</p>
        </div>

        {/* Menu Items */}
        <div className="flex-1 py-4 overflow-y-auto">
          {/* Generation Menu Item - Always Active */}
          <div className="px-6 mb-2">
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">
              {t('navigation.inference')}
            </h3>
          </div>

          {[
            {
              path: "/quick-generate",
              label: t('quickGenerate.generateVoice'),
              icon: "M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z",
            },
            {
              path: "/quick-transcribe",
              label: t('navigation.transcription'),
              icon: "M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z",
            },
          ].map((item) => {
            const isActive = pathname === item.path;
            return (
              <Link
                key={item.path}
                href={item.path}
                className={`flex items-center space-x-3 px-6 py-3 transition-all duration-200 relative ${
                  isActive ? "bg-blue-600 text-white" : "text-gray-300 hover:bg-gray-800 hover:text-white"
                }`}
              >
                {/* Active indicator */}
                {isActive && <div className="absolute left-0 top-0 bottom-0 w-1 bg-blue-400" />}
                <div className={isActive ? "text-white" : "text-gray-400"}>
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d={item.icon} />
                  </svg>
                </div>
                <span className="font-medium text-sm">{item.label}</span>
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
