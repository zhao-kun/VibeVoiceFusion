'use client';

import { useLanguage } from '@/lib/i18n/LanguageContext';
import LiveTranscriptionWorkspace from '@/components/LiveTranscriptionWorkspace';

export default function LiveTranscribePage() {
  const { t } = useLanguage();

  return (
    <LiveTranscriptionWorkspace
      projectId={null}
      recordPath="/quick-transcribe"
      title={t('liveTranscription.pageTitle')}
      subtitle={t('liveTranscription.pageSubtitle')}
      badge={t('quickGenerate.quickMode')}
      badgeClassName="bg-emerald-100 text-emerald-700"
    />
  );
}
