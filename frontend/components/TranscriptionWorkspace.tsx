'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useRouter, useSearchParams } from 'next/navigation';
import { useLanguage } from '@/lib/i18n/LanguageContext';
import { api } from '@/lib/api';
import type { ModelDtype, OffloadingMode, OffloadingPreset } from '@/types/generation';
import type { Transcription } from '@/types/transcription';
import { isTranscriptionActive, segmentSeconds, formatTimestamp } from '@/types/transcription';
import TranscriptionHistory from '@/components/TranscriptionHistory';
import toast from 'react-hot-toast';

const ACCEPTED_AUDIO = '.wav,.mp3,.m4a,.flac,.webm';

// Same GPU layer counts as the backend presets in backend/inference/asr_inference.py
const PRESET_GPU_LAYERS: Record<OffloadingPreset, number> = {
  balanced: 12,
  aggressive: 8,
  extreme: 4,
};

const SPEAKER_COLORS = [
  'bg-blue-100 text-blue-800',
  'bg-purple-100 text-purple-800',
  'bg-amber-100 text-amber-800',
  'bg-emerald-100 text-emerald-800',
  'bg-pink-100 text-pink-800',
  'bg-cyan-100 text-cyan-800',
];

interface TranscriptionWorkspaceProps {
  projectId: string | null;
  basePath: string;
  title: string;
  subtitle: string;
  badge: string;
  badgeClassName: string;
}

export default function TranscriptionWorkspace({
  projectId, basePath, title, subtitle, badge, badgeClassName,
}: TranscriptionWorkspaceProps) {
  const { t } = useLanguage();
  const router = useRouter();
  const searchParams = useSearchParams();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [audioPreviewUrl, setAudioPreviewUrl] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [contextInfo, setContextInfo] = useState('');
  const [modelDtype, setModelDtype] = useState<ModelDtype>('bf16');
  const [maxNewTokens, setMaxNewTokens] = useState(8192);
  const [temperature, setTemperature] = useState(0);
  const [topP, setTopP] = useState(1);
  const [repetitionPenalty, setRepetitionPenalty] = useState(1);
  const [seeds, setSeeds] = useState(() => Math.floor(Math.random() * 1000000));

  const [offloadingEnabled, setOffloadingEnabled] = useState(false);
  const [offloadingMode, setOffloadingMode] = useState<OffloadingMode>('preset');
  const [offloadingPreset, setOffloadingPreset] = useState<OffloadingPreset>('balanced');
  const [manualGpuLayers, setManualGpuLayers] = useState(20);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [current, setCurrent] = useState<Transcription | null>(null);
  const [showRaw, setShowRaw] = useState(false);

  useEffect(() => {
    if (!audioFile) {
      setAudioPreviewUrl(null);
      return;
    }
    const url = URL.createObjectURL(audioFile);
    setAudioPreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [audioFile]);

  // Poll for transcription status
  useEffect(() => {
    if (!current || !isTranscriptionActive(current.status)) return;

    const interval = setInterval(async () => {
      try {
        const updated = await api.getTranscription(projectId, current.request_id);
        setCurrent(updated);
        if (!isTranscriptionActive(updated.status)) {
          clearInterval(interval);
        }
      } catch (err) {
        console.error('Failed to poll transcription status:', err);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [projectId, current?.request_id, current?.status]); // eslint-disable-line react-hooks/exhaustive-deps

  // Load the transcription from the URL, or the one currently running
  useEffect(() => {
    const requestId = searchParams.get('request_id');
    if (requestId) {
      api.getTranscription(projectId, requestId)
        .then(setCurrent)
        .catch(err => console.error('Failed to load transcription:', err));
    } else {
      setCurrent(null);
      api.getCurrentTranscription(projectId)
        .then(response => {
          if (response.transcription) {
            setCurrent(response.transcription);
          }
        })
        .catch(err => console.error('Failed to check current transcription:', err));
    }
  }, [projectId, searchParams]);

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file && (file.type.startsWith('audio/') || file.type.startsWith('video/webm'))) {
      setAudioFile(file);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!audioFile) {
      setError(t('transcription.errorAudioRequired'));
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const offloading = offloadingEnabled ? {
        enabled: true,
        mode: offloadingMode,
        ...(offloadingMode === 'preset'
          ? { preset: offloadingPreset }
          : { num_gpu_layers: manualGpuLayers }
        )
      } : undefined;

      const response = await api.startTranscription(projectId, {
        audio_file: audioFile,
        context_info: contextInfo.trim() || undefined,
        model_dtype: modelDtype,
        max_new_tokens: maxNewTokens,
        temperature,
        top_p: topP,
        repetition_penalty: repetitionPenalty,
        seeds,
        offloading,
      });

      const transcription = await api.getTranscription(projectId, response.request_id);
      setCurrent(transcription);
      router.replace(`${basePath}?request_id=${response.request_id}`);
    } catch (err) {
      setError(err instanceof Error ? err.message : t('transcription.errorGeneric'));
    } finally {
      setLoading(false);
    }
  };

  const handleNew = () => {
    setCurrent(null);
    setShowRaw(false);
    router.replace(basePath);
  };

  const handleSelect = useCallback((requestId: string) => {
    setShowRaw(false);
    router.replace(`${basePath}?request_id=${requestId}`);
  }, [router, basePath]);

  const copyTranscript = async () => {
    if (!current) return;
    const text = current.segments.length > 0
      ? current.segments.map(s => String(s.text).trim()).join('\n')
      : (current.raw_text || '');
    try {
      await navigator.clipboard.writeText(text);
      toast.success(t('transcription.copied'));
    } catch {
      toast.error(t('transcription.errorGeneric'));
    }
  };

  const formatSeconds = (value: number | null | undefined): string =>
    value === null || value === undefined ? '-' : `${value.toFixed(2)}s`;

  const getStatusColor = (status: string): string => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800 border-green-300';
      case 'failed':
        return 'bg-red-100 text-red-800 border-red-300';
      case 'preprocessing':
      case 'inferencing':
        return 'bg-blue-100 text-blue-800 border-blue-300';
      default:
        return 'bg-gray-100 text-gray-800 border-gray-300';
    }
  };

  const speakerColor = (speakerId: number | string | null): string => {
    const index = typeof speakerId === 'number' ? speakerId : parseInt(String(speakerId ?? 0), 10) || 0;
    return SPEAKER_COLORS[Math.abs(index) % SPEAKER_COLORS.length];
  };

  const renderResult = () => {
    if (!current) return null;
    const inProgress = isTranscriptionActive(current.status);
    const isCompleted = current.status === 'completed';
    const details = current.details;

    return (
      <div className={`border-2 rounded-lg p-6 ${getStatusColor(current.status)}`}>
        <div className="flex items-center justify-between mb-4">
          <div className="min-w-0">
            <h2 className="text-xl font-semibold">{t('transcription.currentTranscription')}</h2>
            <p className="text-sm opacity-75 truncate">{current.original_filename}</p>
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            {inProgress && <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-current" />}
            <span className="text-sm font-medium">{t(`transcription.status.${current.status}`)}</span>
          </div>
        </div>

        <div className="space-y-4">
          {/* A live session writes its audio file only when it ends */}
          {!(inProgress && current.source === 'live') && (
            <audio
              controls
              src={api.getTranscriptionAudioUrl(projectId, current.request_id)}
              className="w-full"
            />
          )}

          {current.context_info && (
            <div className="text-sm">
              <span className="font-medium opacity-75">{t('transcription.contextInfo')}: </span>
              <span>{current.context_info}</span>
            </div>
          )}

          {inProgress && (
            <div className="border border-current border-opacity-20 rounded-lg p-4 bg-white bg-opacity-30">
              {/* The final token count is unknown while decoding, so progress is indeterminate */}
              <div className="w-full bg-white bg-opacity-50 rounded-full h-3 overflow-hidden">
                <div className="bg-indigo-500 h-3 rounded-full w-1/3 animate-pulse" />
              </div>
              <div className="flex justify-between text-xs mt-2 opacity-75">
                <span>{t('transcription.generatedTokens', { count: current.generated_tokens })}</span>
                {current.audio_duration !== null && (
                  <span>{t('transcription.audioDuration')}: {formatSeconds(current.audio_duration)}</span>
                )}
              </div>
            </div>
          )}

          {current.status === 'failed' && current.error_message && (
            <div className="p-3 bg-white bg-opacity-50 rounded-lg">
              <p className="text-sm font-medium">{t('transcription.errorTitle')}</p>
              <p className="text-sm mt-1 break-words">{current.error_message}</p>
            </div>
          )}

          {isCompleted && (
            <>
              {current.reach_max_new_tokens && (
                <div className="p-3 bg-amber-50 border border-amber-300 text-amber-800 rounded-lg text-sm">
                  {t('transcription.reachedMaxTokens')}
                </div>
              )}

              <div className="grid grid-cols-3 gap-3 text-sm">
                <div>
                  <p className="opacity-75">{t('transcription.audioDuration')}</p>
                  <p className="font-semibold">{formatSeconds(current.audio_duration)}</p>
                </div>
                <div>
                  <p className="opacity-75">{t('transcription.segments')}</p>
                  <p className="font-semibold">{current.segments.length}</p>
                </div>
                <div>
                  <p className="opacity-75">{t('transcription.tokens')}</p>
                  <p className="font-semibold">{current.generated_tokens}</p>
                </div>
              </div>

              <div className="flex flex-wrap gap-2">
                <a
                  href={api.getTranscriptionDownloadUrl(projectId, current.request_id, 'txt')}
                  className="px-3 py-1.5 bg-white border border-gray-300 text-gray-800 text-sm rounded-lg hover:bg-gray-50"
                >
                  {t('transcription.downloadTxt')}
                </a>
                <a
                  href={api.getTranscriptionDownloadUrl(projectId, current.request_id, 'json')}
                  className="px-3 py-1.5 bg-white border border-gray-300 text-gray-800 text-sm rounded-lg hover:bg-gray-50"
                >
                  {t('transcription.downloadJson')}
                </a>
                <button
                  onClick={copyTranscript}
                  className="px-3 py-1.5 bg-white border border-gray-300 text-gray-800 text-sm rounded-lg hover:bg-gray-50"
                >
                  {t('transcription.copyText')}
                </button>
                <button
                  onClick={() => setShowRaw(v => !v)}
                  className="px-3 py-1.5 bg-white border border-gray-300 text-gray-800 text-sm rounded-lg hover:bg-gray-50"
                >
                  {showRaw ? t('transcription.showSegments') : t('transcription.showRaw')}
                </button>
              </div>

              <div className="bg-white rounded-lg border border-gray-200 text-gray-900 max-h-[28rem] overflow-y-auto">
                {showRaw ? (
                  <pre className="p-4 text-xs whitespace-pre-wrap break-words">{current.raw_text || ''}</pre>
                ) : current.segments.length === 0 ? (
                  <p className="p-4 text-sm text-gray-500">{t('transcription.noSegments')}</p>
                ) : (
                  <ul className="divide-y divide-gray-100">
                    {current.segments.map((segment, index) => (
                      <li key={index} className="p-3 flex gap-3">
                        <div className="flex-shrink-0 w-28">
                          <p className="text-xs font-mono text-gray-500">
                            {formatTimestamp(segmentSeconds(segment.start_time))}
                          </p>
                          <p className="text-xs font-mono text-gray-400">
                            {formatTimestamp(segmentSeconds(segment.end_time))}
                          </p>
                        </div>
                        <div className="flex-1 min-w-0">
                          {segment.speaker_id !== null && segment.speaker_id !== undefined && (
                            <span className={`inline-block px-2 py-0.5 text-xs font-medium rounded-full mb-1 ${speakerColor(segment.speaker_id)}`}>
                              {t('transcription.speaker', { id: String(segment.speaker_id) })}
                            </span>
                          )}
                          <p className="text-sm">{String(segment.text)}</p>
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
              </div>

              {details && (
                <details className="text-xs">
                  <summary className="cursor-pointer font-medium opacity-75">{t('transcription.statistics')}</summary>
                  <div className="grid grid-cols-2 gap-x-4 gap-y-1 mt-2">
                    <span className="opacity-75">{t('transcription.preprocessingTime')}</span>
                    <span>{formatSeconds(details.preprocessing_duration)}</span>
                    <span className="opacity-75">{t('transcription.modelLoadTime')}</span>
                    <span>{formatSeconds(details.model_load_duration)}</span>
                    <span className="opacity-75">{t('transcription.encodeTime')}</span>
                    <span>{formatSeconds(details.encode_time)}</span>
                    <span className="opacity-75">{t('transcription.prefillTime')}</span>
                    <span>{formatSeconds(details.prefill_time)}</span>
                    <span className="opacity-75">{t('transcription.decodeTime')}</span>
                    <span>{formatSeconds(details.decode_time)}</span>
                    <span className="opacity-75">{t('transcription.promptTokens')}</span>
                    <span>{details.prompt_tokens ?? '-'}</span>
                    <span className="opacity-75">{t('transcription.modelDtype')}</span>
                    <span>{current.model_dtype}</span>
                  </div>
                </details>
              )}
            </>
          )}
        </div>
      </div>
    );
  };

  const renderForm = () => (
    <form onSubmit={handleSubmit} className="space-y-5">
      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">{t('transcription.audioFile')}</label>
        <div
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-colors ${
            isDragging ? 'border-blue-500 bg-blue-50' : 'border-gray-300 hover:border-blue-400 hover:bg-gray-50'
          }`}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept={ACCEPTED_AUDIO}
            className="hidden"
            onChange={(e) => setAudioFile(e.target.files?.[0] || null)}
          />
          {audioFile ? (
            <div className="space-y-1">
              <p className="text-sm font-medium text-gray-900 truncate">{audioFile.name}</p>
              <p className="text-xs text-gray-500">{(audioFile.size / (1024 * 1024)).toFixed(2)} MB</p>
            </div>
          ) : (
            <div className="space-y-1">
              <svg className="w-10 h-10 mx-auto text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              <p className="text-sm text-gray-600">{t('transcription.dropOrClick')}</p>
              <p className="text-xs text-gray-400">{t('transcription.supportedFormats')}</p>
            </div>
          )}
        </div>
        {audioFile && audioPreviewUrl && (
          <div className="mt-2 flex items-center gap-2">
            <audio controls src={audioPreviewUrl} className="flex-1 h-10" />
            <button
              type="button"
              onClick={() => {
                setAudioFile(null);
                if (fileInputRef.current) fileInputRef.current.value = '';
              }}
              className="px-3 py-1.5 text-sm border border-gray-300 rounded-lg hover:bg-gray-100"
            >
              {t('common.remove')}
            </button>
          </div>
        )}
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">{t('transcription.contextInfo')}</label>
        <textarea
          value={contextInfo}
          onChange={(e) => setContextInfo(e.target.value)}
          rows={3}
          maxLength={2000}
          placeholder={t('transcription.contextInfoPlaceholder')}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent"
        />
        <p className="text-xs text-gray-500 mt-1">{t('transcription.contextInfoHint')}</p>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">{t('transcription.modelDtype')}</label>
        <select
          value={modelDtype}
          onChange={(e) => setModelDtype(e.target.value as ModelDtype)}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
        >
          <option value="bf16">bf16</option>
          <option value="float8_e4m3fn">float8_e4m3fn</option>
        </select>
        <p className="text-xs text-gray-500 mt-1">{t('transcription.modelDtypeHint')}</p>
      </div>

      <details className="border border-gray-200 rounded-lg">
        <summary className="px-4 py-3 cursor-pointer text-sm font-medium text-gray-700">
          {t('transcription.advancedSettings')}
        </summary>
        <div className="px-4 pb-4 space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-gray-600 mb-1">{t('transcription.maxNewTokens')}</label>
              <input
                type="number"
                min={1}
                max={32768}
                value={maxNewTokens}
                onChange={(e) => setMaxNewTokens(parseInt(e.target.value) || 1)}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-600 mb-1">{t('transcription.seed')}</label>
              <div className="flex gap-2">
                <input
                  type="number"
                  min={0}
                  value={seeds}
                  onChange={(e) => setSeeds(parseInt(e.target.value) || 0)}
                  className="flex-1 min-w-0 px-3 py-2 border border-gray-300 rounded-lg text-sm"
                />
                <button
                  type="button"
                  onClick={() => setSeeds(Math.floor(Math.random() * 1000000))}
                  className="px-2 border border-gray-300 rounded-lg text-sm hover:bg-gray-100"
                  title={t('transcription.randomSeed')}
                >
                  🎲
                </button>
              </div>
            </div>
          </div>

          <div>
            <label className="block text-xs text-gray-600 mb-1">
              {t('transcription.temperature')}: {temperature.toFixed(2)}
            </label>
            <input
              type="range"
              min="0"
              max="2"
              step="0.05"
              value={temperature}
              onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
            <p className="text-xs text-gray-500 mt-1">{t('transcription.temperatureHint')}</p>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-xs text-gray-600 mb-1">
                {t('transcription.topP')}: {topP.toFixed(2)}
              </label>
              <input
                type="range"
                min="0.01"
                max="1"
                step="0.01"
                value={topP}
                onChange={(e) => setTopP(parseFloat(e.target.value))}
                disabled={temperature === 0}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer disabled:opacity-50"
              />
            </div>
            <div>
              <label className="block text-xs text-gray-600 mb-1">
                {t('transcription.repetitionPenalty')}: {repetitionPenalty.toFixed(2)}
              </label>
              <input
                type="range"
                min="0.5"
                max="2"
                step="0.05"
                value={repetitionPenalty}
                onChange={(e) => setRepetitionPenalty(parseFloat(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
            </div>
          </div>

          <div className="pt-3 border-t border-gray-200">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={offloadingEnabled}
                onChange={(e) => setOffloadingEnabled(e.target.checked)}
                className="w-4 h-4 text-blue-500 rounded focus:ring-2 focus:ring-blue-500"
              />
              <span className="text-sm font-medium text-gray-700">{t('generation.enableOffloading')}</span>
            </label>

            {offloadingEnabled && (
              <div className="mt-3 space-y-3">
                <div className="space-y-2">
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="radio"
                      value="preset"
                      checked={offloadingMode === 'preset'}
                      onChange={(e) => setOffloadingMode(e.target.value as OffloadingMode)}
                      className="w-4 h-4 text-blue-500 focus:ring-2 focus:ring-blue-500"
                    />
                    <span className="text-sm text-gray-700">{t('generation.presetRecommended')}</span>
                  </label>
                  <label className="flex items-center gap-2 cursor-pointer">
                    <input
                      type="radio"
                      value="manual"
                      checked={offloadingMode === 'manual'}
                      onChange={(e) => setOffloadingMode(e.target.value as OffloadingMode)}
                      className="w-4 h-4 text-blue-500 focus:ring-2 focus:ring-blue-500"
                    />
                    <span className="text-sm text-gray-700">{t('generation.manualAdvanced')}</span>
                  </label>
                </div>

                {offloadingMode === 'preset' && (
                  <select
                    value={offloadingPreset}
                    onChange={(e) => setOffloadingPreset(e.target.value as OffloadingPreset)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm"
                  >
                    {(Object.keys(PRESET_GPU_LAYERS) as OffloadingPreset[]).map(preset => (
                      <option key={preset} value={preset}>
                        {t(`generation.${preset}`)} ({t('generation.gpuLayers')}: {PRESET_GPU_LAYERS[preset]})
                      </option>
                    ))}
                  </select>
                )}

                {offloadingMode === 'manual' && (
                  <div>
                    <label className="block text-xs text-gray-600 mb-1">
                      {t('generation.gpuLayers')}: {manualGpuLayers}
                    </label>
                    <input
                      type="range"
                      min="1"
                      max="28"
                      value={manualGpuLayers}
                      onChange={(e) => setManualGpuLayers(parseInt(e.target.value))}
                      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                    />
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </details>

      {error && (
        <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-sm text-red-800">{error}</p>
        </div>
      )}

      <button
        type="submit"
        disabled={loading || !audioFile}
        className="w-full px-4 py-3 bg-blue-500 text-white font-medium rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
      >
        {loading ? (
          <span className="flex items-center justify-center gap-2">
            <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white" />
            {t('transcription.starting')}
          </span>
        ) : (
          t('transcription.start')
        )}
      </button>
    </form>
  );

  return (
    <div className="h-full flex flex-col">
      <header className="bg-white border-b border-gray-200 px-6 py-4">
        <div className="flex items-center space-x-2 mb-1">
          <h1 className="text-2xl font-bold text-gray-900">{title}</h1>
          <span className={`px-3 py-1 text-xs font-medium rounded-full ${badgeClassName}`}>{badge}</span>
        </div>
        <p className="text-sm text-gray-500">{subtitle}</p>
      </header>

      <div className="flex-1 overflow-hidden bg-gray-50">
        <div className="h-full grid grid-cols-2 gap-6 p-6">
          <div className="bg-white rounded-lg shadow-sm p-6 overflow-hidden flex flex-col">
            <TranscriptionHistory
              projectId={projectId}
              onSelect={handleSelect}
              currentId={current?.request_id}
              currentStatus={current?.status}
            />
          </div>

          <div className="flex flex-col gap-6 overflow-y-auto">
            {current && renderResult()}

            {current && !isTranscriptionActive(current.status) && (
              <button
                onClick={handleNew}
                className="w-full px-4 py-3 bg-blue-500 text-white text-sm font-medium rounded-lg hover:bg-blue-600 transition-colors flex items-center justify-center gap-2 shadow-sm"
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                </svg>
                {t('transcription.newTranscription')}
              </button>
            )}

            {!current && (
              <div className="bg-white rounded-lg shadow-sm p-6">
                {renderForm()}
              </div>
            )}

            {!current && (
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <h3 className="text-sm font-semibold text-blue-900 mb-2">{t('transcription.howItWorks')}</h3>
                <ul className="text-sm text-blue-800 space-y-1">
                  <li>{t('transcription.step1')}</li>
                  <li>{t('transcription.step2')}</li>
                  <li>{t('transcription.step3')}</li>
                  <li>{t('transcription.step4')}</li>
                </ul>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
