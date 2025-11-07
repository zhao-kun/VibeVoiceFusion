'use client';

import React from 'react';
import { useGeneration } from '@/lib/GenerationContext';
import { useProject } from '@/lib/ProjectContext';
import { useLanguage } from '@/lib/i18n/LanguageContext';
import { api } from '@/lib/api';
import { InferencePhase, getOffloadingConfig, getOffloadingMetrics } from '@/types/generation';

function CurrentGeneration() {
  const { currentGeneration } = useGeneration();
  const { currentProject } = useProject();
  const { t } = useLanguage();

  if (!currentGeneration) {
    return null;
  }

  const getStatusColor = (status: InferencePhase): string => {
    switch (status) {
      case InferencePhase.COMPLETED:
        return 'bg-green-100 text-green-800 border-green-300';
      case InferencePhase.FAILED:
        return 'bg-red-100 text-red-800 border-red-300';
      case InferencePhase.PENDING:
        return 'bg-gray-100 text-gray-800 border-gray-300';
      case InferencePhase.PREPROCESSING:
      case InferencePhase.INFERENCING:
      case InferencePhase.SAVING_AUDIO:
        return 'bg-blue-100 text-blue-800 border-blue-300';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-300';
    }
  };

  const getStatusLabel = (status: InferencePhase): string => {
    switch (status) {
      case InferencePhase.COMPLETED:
        return t('generation.statusCompleted');
      case InferencePhase.FAILED:
        return t('generation.statusFailed');
      case InferencePhase.PENDING:
        return t('generation.statusPending');
      case InferencePhase.PREPROCESSING:
        return t('generation.statusPreprocessing');
      case InferencePhase.INFERENCING:
        return t('generation.statusInferencing');
      case InferencePhase.SAVING_AUDIO:
        return t('generation.statusSavingAudio');
      default:
        return status;
    }
  };

  const isActive = ['pending', 'preprocessing', 'inferencing', 'saving_audio'].includes(
    currentGeneration.status
  );

  // Extract progress info from details if available
  const currentStep = currentGeneration.details?.current || null;
  const totalSteps = currentGeneration.details?.total_step || null;
  const progressPercentage = currentStep && totalSteps
    ? (currentStep / totalSteps) * 100
    : currentGeneration.percentage;

  // Render phase-specific details
  const renderPhaseDetails = () => {
    const details = currentGeneration.details;

    // PREPROCESSING phase
    if (currentGeneration.status === InferencePhase.PREPROCESSING && details) {
      return (
        <div className="space-y-3">
          <label className="text-sm font-medium opacity-75 block">{t('generation.preprocessingInformation')}</label>

          {details.unique_speaker_names && (
            <div className="bg-white bg-opacity-50 rounded p-3">
              <p className="text-xs font-medium mb-1">{t('generation.speakers')} ({details.unique_speaker_names.length})</p>
              <div className="flex flex-wrap gap-1">
                {details.unique_speaker_names.map((speaker: string, idx: number) => (
                  <span key={idx} className="px-2 py-1 bg-blue-100 text-blue-800 rounded text-xs">
                    {speaker}
                  </span>
                ))}
              </div>
            </div>
          )}

          {details.scripts && (
            <div className="bg-white bg-opacity-50 rounded p-3">
              <p className="text-xs font-medium mb-1">{t('generation.dialogLines')}: {details.scripts.length}</p>
            </div>
          )}
        </div>
      );
    }

    // SAVING_AUDIO or COMPLETED phase
    if ((currentGeneration.status === InferencePhase.SAVING_AUDIO ||
         currentGeneration.status === InferencePhase.COMPLETED) && details) {
      const audioUrl = currentProject && currentGeneration.output_filename
        ? api.getGenerationDownloadUrl(currentProject.id, currentGeneration.request_id)
        : null;

      return (
        <div className="space-y-3">
          {/* Audio Player (for completed generations) */}
          {currentGeneration.status === InferencePhase.COMPLETED && audioUrl && (
            <div className="bg-white bg-opacity-90 rounded-lg p-4 border-2 border-white">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 20 20">
                    <path d="M18 3a1 1 0 00-1.196-.98l-10 2A1 1 0 006 5v9.114A4.369 4.369 0 005 14c-1.657 0-3 .895-3 2s1.343 2 3 2 3-.895 3-2V7.82l8-1.6v5.894A4.37 4.37 0 0015 12c-1.657 0-3 .895-3 2s1.343 2 3 2 3-.895 3-2V3z" />
                  </svg>
                  <label className="text-sm font-semibold">{t('generation.generatedAudio')}</label>
                </div>
                <a
                  href={audioUrl + '?download=true'}
                  download={currentGeneration.output_filename}
                  className="px-3 py-1 text-xs bg-current text-white rounded-lg hover:opacity-80 transition-opacity flex items-center gap-1"
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                  </svg>
                  {t('generation.download')}
                </a>
              </div>
              <audio
                controls
                className="w-full"
                preload="metadata"
              >
                <source src={audioUrl} type="audio/wav" />
                Your browser does not support the audio element.
              </audio>
              <div className="mt-2 flex items-center gap-4 text-xs opacity-75">
                {details.audio_duration_seconds && (
                  <span>{t('generation.duration')}: {details.audio_duration_seconds.toFixed(2)}s</span>
                )}
                {currentGeneration.output_filename && (
                  <span className="font-mono truncate">{currentGeneration.output_filename}</span>
                )}
              </div>
            </div>
          )}

          <label className="text-sm font-medium opacity-75 block">{t('generation.generationStatistics')}</label>

          {/* Performance Summary */}
          {(details.generation_time !== undefined || details.audio_duration_seconds !== undefined || details.real_time_factor !== undefined) && (
            <div className="bg-white bg-opacity-90 rounded p-4">
              <p className="text-xs font-medium mb-3 opacity-75">⚡ {t('generation.performanceMetrics')}</p>
              <div className="grid grid-cols-3 gap-3">
                {details.generation_time !== undefined && (
                  <div className="text-center">
                    <p className="text-2xl font-bold">{details.generation_time.toFixed(2)}s</p>
                    <p className="text-xs opacity-75 mt-1">{t('generation.generationTime')}</p>
                  </div>
                )}
                {details.audio_duration_seconds !== undefined && (
                  <div className="text-center">
                    <p className="text-2xl font-bold">{details.audio_duration_seconds.toFixed(2)}s</p>
                    <p className="text-xs opacity-75 mt-1">{t('generation.audioDuration')}</p>
                  </div>
                )}
                {details.real_time_factor !== undefined && (
                  <div className="text-center">
                    <p className="text-2xl font-bold">{details.real_time_factor.toFixed(2)}×</p>
                    <p className="text-xs opacity-75 mt-1">{t('generation.realTimeFactor')}</p>
                  </div>
                )}
              </div>
              {details.number_of_segments !== undefined && (
                <div className="mt-3 pt-3 border-t border-current border-opacity-20 text-center">
                  <p className="text-xs opacity-75">
                    {t('generation.generatedSegments').replace('{count}', details.number_of_segments.toString())}
                  </p>
                </div>
              )}
            </div>
          )}

          {/* Token Statistics */}
          <div className="bg-white bg-opacity-90 rounded p-4">
            <p className="text-xs font-medium mb-3 opacity-75">🔢 {t('generation.tokenStatistics')}</p>
            <div className="grid grid-cols-3 gap-3">
              {details.prefilling_tokens !== undefined && (
                <div className="text-center">
                  <p className="text-xl font-bold">{details.prefilling_tokens.toLocaleString()}</p>
                  <p className="text-xs opacity-75 mt-1">{t('generation.prefilling')}</p>
                </div>
              )}
              {details.generated_tokens !== undefined && (
                <div className="text-center">
                  <p className="text-xl font-bold">{details.generated_tokens.toLocaleString()}</p>
                  <p className="text-xs opacity-75 mt-1">{t('generation.generated')}</p>
                </div>
              )}
              {details.total_tokens !== undefined && (
                <div className="text-center">
                  <p className="text-xl font-bold">{details.total_tokens.toLocaleString()}</p>
                  <p className="text-xs opacity-75 mt-1">{t('generation.totalTokens')}</p>
                </div>
              )}
            </div>
          </div>

          {/* Speaker Information */}
          {details.unique_speaker_names && (
            <div className="bg-white bg-opacity-90 rounded p-3">
              <p className="text-xs font-medium mb-2 opacity-75">🎤 {t('generation.speakersUsed').replace('{count}', details.unique_speaker_names.length.toString())}</p>
              <div className="flex flex-wrap gap-1">
                {details.unique_speaker_names.map((speaker: string, idx: number) => (
                  <span key={idx} className="px-3 py-1.5 bg-white rounded-full text-xs font-medium border border-current border-opacity-30">
                    {speaker}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Output File Path */}
          {details.output_audio_path && (
            <div className="bg-white bg-opacity-90 rounded p-3">
              <p className="text-xs font-medium mb-1 opacity-75">📁 {t('generation.outputPath')}</p>
              <p className="text-xs font-mono break-all opacity-75">{details.output_audio_path}</p>
            </div>
          )}
        </div>
      );
    }

    // FAILED phase
    if (currentGeneration.status === InferencePhase.FAILED && details) {
      return (
        <div className="space-y-3">
          <label className="text-sm font-medium opacity-75 block">{t('generation.errorInformation')}</label>
          <div className="bg-red-50 bg-opacity-50 rounded p-3">
            <pre className="text-xs whitespace-pre-wrap text-red-900">
              {details.error || JSON.stringify(details, null, 2)}
            </pre>
          </div>
        </div>
      );
    }

    // Fallback for any other details
    if (details && Object.keys(details).length > 0) {
      return (
        <div className="space-y-3">
          <label className="text-sm font-medium opacity-75 block">{t('generation.additionalDetails')}</label>
          <div className="bg-white bg-opacity-50 rounded p-3 max-h-48 overflow-y-auto">
            <pre className="text-xs whitespace-pre-wrap">
              {JSON.stringify(details, null, 2)}
            </pre>
          </div>
        </div>
      );
    }

    return null;
  };

  return (
    <div className={`border-2 rounded-lg p-6 ${getStatusColor(currentGeneration.status)}`}>
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-semibold">{t('generation.currentGeneration')}</h2>
        {isActive && (
          <div className="flex items-center gap-2">
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-current"></div>
            <span className="text-sm font-medium">{t('generation.status')}...</span>
          </div>
        )}
      </div>

      <div className="space-y-4">
        {/* Status */}
        <div>
          <label className="text-sm font-medium opacity-75">{t('generation.status')}</label>
          <p className="text-lg font-semibold">{getStatusLabel(currentGeneration.status)}</p>
        </div>

        {/* Session Name */}
        <div>
          <label className="text-sm font-medium opacity-75 block">{t('generation.sessionName')}</label>
          {(currentGeneration.session_name === t('session.deleted') || currentGeneration.session_name === t('session.unknown')) ? (
            <p className="text-sm text-white opacity-75 italic">{currentGeneration.session_name}</p>
          ) : (
            <a
              href={`/voice-editor?session=${currentGeneration.session_id}`}
              className="text-sm hover:underline cursor-pointer block"
              title={t('generation.sessionId') + ': ' + currentGeneration.session_id}
            >
              {currentGeneration.session_name}
            </a>
          )}
        </div>

        {/* Request ID */}
        <div>
          <label className="text-sm font-medium opacity-75">{t('generation.requestId')}</label>
          <p className="text-sm font-mono">{currentGeneration.request_id}</p>
        </div>

        {/* Progress Bar (if inferencing) */}
        {isActive && progressPercentage !== null && (
          <div>
            <label className="text-sm font-medium opacity-75 mb-2 block">{t('generation.progress')}</label>
            <div className="w-full bg-white bg-opacity-50 rounded-full h-4">
              <div
                className="bg-current h-4 rounded-full transition-all duration-300 flex items-center justify-center"
                style={{ width: `${Math.min(100, Math.max(0, progressPercentage))}%` }}
              >
                <span className="text-xs font-medium text-white px-2">
                  {progressPercentage.toFixed(1)}%
                </span>
              </div>
            </div>
            {currentStep !== null && totalSteps !== null && (
              <p className="text-xs mt-1 opacity-75">
                {t('generation.status')} {currentStep} {t('generation.of')} {totalSteps}
              </p>
            )}
          </div>
        )}

        {/* Model Settings */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="text-sm font-medium opacity-75">{t('generation.modelType')}</label>
            <p className="text-sm">{currentGeneration.model_dtype}</p>
          </div>
          <div>
            <label className="text-sm font-medium opacity-75">{t('generation.cfgScale')}</label>
            <p className="text-sm">{currentGeneration.cfg_scale ?? 'N/A'}</p>
          </div>
          <div>
            <label className="text-sm font-medium opacity-75">{t('generation.seeds')}</label>
            <p className="text-sm">{currentGeneration.seeds}</p>
          </div>
          <div>
            <label className="text-sm font-medium opacity-75">{t('generation.attention')}</label>
            <p className="text-sm">{currentGeneration.attn_implementation ?? 'N/A'}</p>
          </div>
        </div>

        {/* Offloading Status */}
        {(() => {
          const offloadingConfig = getOffloadingConfig(currentGeneration);
          const offloadingMetrics = getOffloadingMetrics(currentGeneration);

          if (offloadingConfig) {
            return (
              <div className="border border-current border-opacity-20 rounded-lg p-4 bg-white bg-opacity-30">
                <div className="flex items-center gap-2 mb-2">
                  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5">
                    <path fillRule="evenodd" d="M7.84 1.804A1 1 0 018.82 1h2.36a1 1 0 01.98.804l.331 1.652a6.993 6.993 0 011.929 1.115l1.598-.54a1 1 0 011.186.447l1.18 2.044a1 1 0 01-.205 1.251l-1.267 1.113a7.047 7.047 0 010 2.228l1.267 1.113a1 1 0 01.206 1.25l-1.18 2.045a1 1 0 01-1.187.447l-1.598-.54a6.993 6.993 0 01-1.929 1.115l-.33 1.652a1 1 0 01-.98.804H8.82a1 1 0 01-.98-.804l-.331-1.652a6.993 6.993 0 01-1.929-1.115l-1.598.54a1 1 0 01-1.186-.447l-1.18-2.044a1 1 0 01.205-1.251l1.267-1.114a7.05 7.05 0 010-2.227L1.821 7.773a1 1 0 01-.206-1.25l1.18-2.045a1 1 0 011.187-.447l1.598.54A6.993 6.993 0 017.51 3.456l.33-1.652zM10 13a3 3 0 100-6 3 3 0 000 6z" clipRule="evenodd" />
                  </svg>
                  <span className="text-sm font-medium">{t('generation.offloading')}: {t('generation.offloadingEnabled')}</span>
                </div>
                <div className="text-sm ml-7">
                  {offloadingConfig.mode === 'preset' && offloadingConfig.preset && (
                    <p className="capitalize opacity-75">
                      {t(`generation.offloading${offloadingConfig.preset.charAt(0).toUpperCase() + offloadingConfig.preset.slice(1)}`)}
                    </p>
                  )}
                  {offloadingConfig.mode === 'manual' && offloadingConfig.num_gpu_layers !== undefined && (
                    <p className="opacity-75">
                      {offloadingConfig.num_gpu_layers} {t('generation.gpuLayers')}
                    </p>
                  )}
                  {offloadingMetrics && (
                    <p className="mt-1 opacity-75">
                      {t('generation.transferOverhead')}: {offloadingMetrics.overhead_percentage.toFixed(1)}%
                    </p>
                  )}
                </div>
              </div>
            );
          }
          return null;
        })()}

        {/* Phase-Specific Details */}
        {renderPhaseDetails()}

        {/* Timestamps */}
        <div className="text-xs opacity-75 pt-2 border-t border-current border-opacity-20">
          <p>{t('generation.created')}: {new Date(currentGeneration.created_at).toLocaleString()}</p>
          <p>{t('generation.updated')}: {new Date(currentGeneration.updated_at).toLocaleString()}</p>
        </div>
      </div>
    </div>
  );
}

// Export with React.memo to prevent unnecessary re-renders
export default React.memo(CurrentGeneration);
