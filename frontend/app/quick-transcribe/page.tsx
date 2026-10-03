'use client';

import { Suspense } from 'react';
import { useLanguage } from '@/lib/i18n/LanguageContext';
import TranscriptionWorkspace from '@/components/TranscriptionWorkspace';

function QuickTranscribeContent() {
  const { t } = useLanguage();

  return (
    <TranscriptionWorkspace
      projectId={null}
      basePath="/quick-transcribe"
      title={t('transcription.pageTitle')}
      subtitle={t('transcription.pageSubtitle')}
      badge={t('quickGenerate.quickMode')}
      badgeClassName="bg-emerald-100 text-emerald-700"
    />
  );
}

export default function QuickTranscribePage() {
  return (
    <Suspense fallback={
      <div className="h-full flex items-center justify-center bg-gray-50">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600" />
      </div>
    }>
      <QuickTranscribeContent />
    </Suspense>
  );
}
