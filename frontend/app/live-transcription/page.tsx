'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useProject } from '@/lib/ProjectContext';
import { useLanguage } from '@/lib/i18n/LanguageContext';
import LiveTranscriptionWorkspace from '@/components/LiveTranscriptionWorkspace';

export default function LiveTranscriptionPage() {
  const router = useRouter();
  const { currentProject, loading } = useProject();
  const { t } = useLanguage();

  // Redirect to home page if no project is selected (after loading completes)
  useEffect(() => {
    if (!loading && !currentProject) {
      router.push('/');
    }
  }, [loading, currentProject, router]);

  if (!loading && currentProject) {
    return (
      <LiveTranscriptionWorkspace
        key={currentProject.id}
        projectId={currentProject.id}
        recordPath="/transcription"
        title={t('liveTranscription.pageTitle')}
        subtitle={t('liveTranscription.projectPageSubtitle')}
        badge={currentProject.name}
        badgeClassName="bg-blue-100 text-blue-700"
      />
    );
  }

  return (
    <div className="h-full flex flex-col">
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <h1 className="text-2xl font-bold text-gray-900">{t('liveTranscription.pageTitle')}</h1>
        <p className="text-sm text-gray-500 mt-1">{t('liveTranscription.projectPageSubtitle')}</p>
      </header>

      <div className="flex-1 flex items-center justify-center bg-gray-50">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
          <p className="text-gray-500">
            {loading ? t('generation.loadingProject') : t('generation.redirecting')}
          </p>
        </div>
      </div>
    </div>
  );
}
