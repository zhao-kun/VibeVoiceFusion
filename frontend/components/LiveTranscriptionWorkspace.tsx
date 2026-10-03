'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';
import Link from 'next/link';
import { useLanguage } from '@/lib/i18n/LanguageContext';
import { api } from '@/lib/api';
import type { OffloadingPreset } from '@/types/generation';
import type {
  LiveSessionStatus,
  RealtimeServerEvent,
  TranscriptionSegment,
} from '@/types/transcription';
import { segmentSeconds, formatTimestamp } from '@/types/transcription';
import toast from 'react-hot-toast';

const SAMPLE_RATE = 24000;
const WORKLET_URL = '/pcm16-worklet.js';
const API_KEY_SUBPROTOCOL_PREFIX = 'openai-insecure-api-key.';

// Model names of the OpenAI-compatible API, see ASR_MODEL_MAPPING in backend/services/openai_compat_service.py
const MODELS = [
  { name: 'vibevoice-asr', dtype: 'bf16' },
  { name: 'vibevoice-asr-fp8', dtype: 'float8_e4m3fn' },
];

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

type Phase = 'idle' | 'connecting' | 'live' | 'stopping' | 'done' | 'error';

interface LiveItem {
  id: string;
  text: string;
  segment: TranscriptionSegment | null;
  completed: boolean;
}

interface Hypothesis {
  text: string;
  segments: TranscriptionSegment[];
}

interface WorkletChunk {
  pcm: ArrayBuffer;
  level: number;
  final: boolean;
}

interface LiveTranscriptionWorkspaceProps {
  projectId: string | null;
  recordPath: string;
  title: string;
  subtitle: string;
  badge: string;
  badgeClassName: string;
}

function toBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = '';
  const step = 0x8000;
  for (let i = 0; i < bytes.length; i += step) {
    binary += String.fromCharCode(...bytes.subarray(i, i + step));
  }
  return btoa(binary);
}

function formatElapsed(seconds: number): string {
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
}

export default function LiveTranscriptionWorkspace({
  projectId,
  recordPath,
  title,
  subtitle,
  badge,
  badgeClassName,
}: LiveTranscriptionWorkspaceProps) {
  const { t } = useLanguage();

  const [model, setModel] = useState(MODELS[0].name);
  const [prompt, setPrompt] = useState('');
  const [offloadingEnabled, setOffloadingEnabled] = useState(false);
  const [offloadingPreset, setOffloadingPreset] = useState<OffloadingPreset>('balanced');
  const [apiKey, setApiKey] = useState('');

  const [phase, setPhase] = useState<Phase>('idle');
  const [serverStatus, setServerStatus] = useState<LiveSessionStatus | null>(null);
  const [items, setItems] = useState<LiveItem[]>([]);
  const [hypothesis, setHypothesis] = useState<Hypothesis>({ text: '', segments: [] });
  const [requestId, setRequestId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const [level, setLevel] = useState(0);

  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const workletRef = useRef<AudioWorkletNode | null>(null);
  const pendingRef = useRef<string[]>([]);
  const sentSamplesRef = useRef(0);
  const finishSentRef = useRef(false);
  const finalStatusRef = useRef<LiveSessionStatus | null>(null);
  const errorRef = useRef<string | null>(null);
  const transcriptEndRef = useRef<HTMLDivElement | null>(null);

  const isActive = phase === 'connecting' || phase === 'live' || phase === 'stopping';

  const releaseAudio = useCallback(() => {
    if (workletRef.current) {
      workletRef.current.port.onmessage = null;
      workletRef.current.disconnect();
      workletRef.current = null;
    }
    streamRef.current?.getTracks().forEach(track => track.stop());
    streamRef.current = null;
    if (audioContextRef.current) {
      audioContextRef.current.close().catch(() => {});
      audioContextRef.current = null;
    }
    setLevel(0);
  }, []);

  useEffect(() => {
    return () => {
      // Closing the socket still lets the backend transcribe and save what it received
      wsRef.current?.close();
      wsRef.current = null;
      releaseAudio();
    };
  }, [releaseAudio]);

  useEffect(() => {
    transcriptEndRef.current?.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }, [items, hypothesis]);

  const sendEvent = useCallback((event: Record<string, unknown>) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(event));
    }
  }, []);

  const sendFinish = useCallback(() => {
    if (finishSentRef.current) return;
    finishSentRef.current = true;
    sendEvent({ type: 'vibevoice.session.finish' });
  }, [sendEvent]);

  const reportError = useCallback((message: string) => {
    errorRef.current = message;
    setError(message);
  }, []);

  const errorMessage = useCallback((code: string | undefined, message: string): string => {
    if (code === 'server_busy') return t('liveTranscription.errorBusy');
    if (code === 'invalid_api_key') return t('liveTranscription.errorApiKey');
    return message || t('transcription.errorGeneric');
  }, [t]);

  const handleServerEvent = useCallback((event: RealtimeServerEvent) => {
    switch (event.type) {
      case 'input_audio_buffer.committed':
        setItems(prev => prev.some(item => item.id === event.item_id)
          ? prev
          : [...prev, { id: event.item_id, text: '', segment: null, completed: false }]);
        break;
      case 'conversation.item.input_audio_transcription.delta':
        setItems(prev => prev.some(item => item.id === event.item_id)
          ? prev.map(item => item.id === event.item_id ? { ...item, text: item.text + event.delta } : item)
          : [...prev, { id: event.item_id, text: event.delta, segment: null, completed: false }]);
        break;
      case 'conversation.item.input_audio_transcription.completed': {
        const completed: LiveItem = {
          id: event.item_id,
          text: event.transcript,
          segment: event.segment ?? null,
          completed: true,
        };
        setItems(prev => prev.some(item => item.id === event.item_id)
          ? prev.map(item => item.id === event.item_id ? completed : item)
          : [...prev, completed]);
        break;
      }
      case 'vibevoice.transcription.hypothesis':
        setHypothesis({ text: event.text, segments: event.segments });
        break;
      case 'vibevoice.session.status':
        finalStatusRef.current = event.status;
        setServerStatus(event.status);
        if (event.request_id) {
          setRequestId(event.request_id);
        }
        if (event.status === 'completed') {
          setHypothesis({ text: '', segments: [] });
        }
        if (event.status === 'failed') {
          reportError(event.error || t('transcription.errorGeneric'));
        }
        break;
      case 'error':
        reportError(errorMessage(event.error.code, event.error.message));
        break;
      default:
        break;
    }
  }, [errorMessage, reportError, t]);

  const handleChunk = useCallback((chunk: WorkletChunk) => {
    setLevel(chunk.level);
    if (chunk.pcm.byteLength > 0 && !finishSentRef.current) {
      const audio = toBase64(chunk.pcm);
      const ws = wsRef.current;
      if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'input_audio_buffer.append', audio }));
      } else if (ws && ws.readyState === WebSocket.CONNECTING) {
        pendingRef.current.push(audio);
      }
      sentSamplesRef.current += chunk.pcm.byteLength / 2;
      setElapsed(sentSamplesRef.current / SAMPLE_RATE);
    }
    if (chunk.final) {
      sendFinish();
      releaseAudio();
    }
  }, [releaseAudio, sendFinish]);

  const resetSession = () => {
    setItems([]);
    setHypothesis({ text: '', segments: [] });
    setServerStatus(null);
    setRequestId(null);
    setError(null);
    setElapsed(0);
    errorRef.current = null;
    finalStatusRef.current = null;
    finishSentRef.current = false;
    sentSamplesRef.current = 0;
    pendingRef.current = [];
  };

  const startSession = async () => {
    resetSession();
    setPhase('connecting');

    if (!navigator.mediaDevices?.getUserMedia) {
      reportError(t('liveTranscription.errorInsecureContext'));
      setPhase('error');
      return;
    }

    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
      });
    } catch (err) {
      const name = err instanceof DOMException ? err.name : '';
      if (name === 'NotAllowedError' || name === 'SecurityError') {
        reportError(t('liveTranscription.errorPermission'));
      } else if (name === 'NotFoundError') {
        reportError(t('liveTranscription.errorNoMicrophone'));
      } else {
        reportError(t('liveTranscription.errorMicrophone'));
      }
      setPhase('error');
      return;
    }
    streamRef.current = stream;

    try {
      const context = new AudioContext();
      audioContextRef.current = context;
      // Created after an await, so some browsers no longer count it as part of the click gesture
      if (context.state === 'suspended') {
        await context.resume();
      }
      await context.audioWorklet.addModule(WORKLET_URL);
      const source = context.createMediaStreamSource(stream);
      const node = new AudioWorkletNode(context, 'pcm16-processor', {
        processorOptions: { targetRate: SAMPLE_RATE, chunkSeconds: 0.1 },
      });
      node.port.onmessage = (event: MessageEvent<WorkletChunk>) => handleChunk(event.data);
      source.connect(node);
      // The worklet writes no output; connecting it keeps the graph pulling audio in every browser
      node.connect(context.destination);
      workletRef.current = node;
    } catch (err) {
      console.error('Failed to start audio capture:', err);
      releaseAudio();
      reportError(t('liveTranscription.errorMicrophone'));
      setPhase('error');
      return;
    }

    const protocols = ['realtime'];
    if (apiKey.trim()) {
      protocols.push(`${API_KEY_SUBPROTOCOL_PREFIX}${apiKey.trim()}`);
    }

    let ws: WebSocket;
    try {
      ws = new WebSocket(
        api.getRealtimeTranscriptionUrl(projectId, offloadingEnabled ? offloadingPreset : undefined),
        protocols,
      );
    } catch (err) {
      console.error('Failed to open realtime connection:', err);
      releaseAudio();
      reportError(t('liveTranscription.errorConnection'));
      setPhase('error');
      return;
    }
    wsRef.current = ws;

    ws.onopen = () => {
      ws.send(JSON.stringify({
        type: 'session.update',
        session: {
          type: 'transcription',
          audio: {
            input: {
              format: { type: 'audio/pcm', rate: SAMPLE_RATE },
              transcription: { model, prompt: prompt.trim() },
              turn_detection: null,
            },
          },
        },
      }));
      pendingRef.current.forEach(audio => ws.send(JSON.stringify({ type: 'input_audio_buffer.append', audio })));
      pendingRef.current = [];
      setPhase(current => (current === 'connecting' ? 'live' : current));
    };

    ws.onmessage = (message: MessageEvent) => {
      if (typeof message.data !== 'string') return;
      try {
        handleServerEvent(JSON.parse(message.data) as RealtimeServerEvent);
      } catch (err) {
        console.error('Invalid realtime event:', err);
      }
    };

    ws.onclose = () => {
      if (wsRef.current === ws) {
        wsRef.current = null;
      }
      releaseAudio();
      if (finalStatusRef.current === 'completed' && !errorRef.current) {
        setPhase('done');
        return;
      }
      if (!errorRef.current) {
        reportError(t('liveTranscription.errorConnectionLost'));
      }
      setPhase('error');
    };
  };

  const stopSession = () => {
    if (!isActive || phase === 'stopping') return;
    setPhase('stopping');
    const ws = wsRef.current;
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      ws?.close();
      releaseAudio();
      return;
    }
    if (workletRef.current) {
      // The worklet answers with its partial chunk (final=true), which then sends the finish event
      workletRef.current.port.postMessage('flush');
    } else {
      sendFinish();
    }
  };

  const transcriptText = () => {
    const parts = items.map(item => item.text.trim()).filter(Boolean);
    if (hypothesis.text) parts.push(hypothesis.text);
    return parts.join('\n');
  };

  const copyTranscript = async () => {
    try {
      await navigator.clipboard.writeText(transcriptText());
      toast.success(t('transcription.copied'));
    } catch {
      toast.error(t('transcription.errorGeneric'));
    }
  };

  const speakerColor = (speakerId: number | string): string => {
    const index = typeof speakerId === 'number' ? speakerId : parseInt(String(speakerId), 10) || 0;
    return SPEAKER_COLORS[Math.abs(index) % SPEAKER_COLORS.length];
  };

  const statusLabel = (): string => {
    if (phase === 'connecting') return t('liveTranscription.status.connecting');
    if (phase === 'done') return t('liveTranscription.status.completed');
    if (phase === 'error') return t('liveTranscription.status.failed');
    if (serverStatus) return t(`liveTranscription.status.${serverStatus}`);
    return t('liveTranscription.status.connecting');
  };

  const statusDotClass = (): string => {
    if (phase === 'error') return 'bg-red-500';
    if (phase === 'done') return 'bg-green-500';
    if (serverStatus === 'listening' && phase === 'live') return 'bg-red-500 animate-pulse';
    return 'bg-amber-500 animate-pulse';
  };

  const renderSegmentRow = (
    key: string,
    segment: TranscriptionSegment | null,
    content: React.ReactNode,
    tentative: boolean,
  ) => (
    <li key={key} className={`p-3 flex gap-3 ${tentative ? 'bg-gray-50' : ''}`}>
      <div className="flex-shrink-0 w-28">
        <p className="text-xs font-mono text-gray-500">{formatTimestamp(segmentSeconds(segment?.start_time))}</p>
        <p className="text-xs font-mono text-gray-400">{formatTimestamp(segmentSeconds(segment?.end_time))}</p>
      </div>
      <div className="flex-1 min-w-0">
        {segment && segment.speaker_id !== null && segment.speaker_id !== undefined && (
          <span className={`inline-block px-2 py-0.5 text-xs font-medium rounded-full mb-1 ${speakerColor(segment.speaker_id)}`}>
            {t('transcription.speaker', { id: String(segment.speaker_id) })}
          </span>
        )}
        <p className="text-sm">{content}</p>
      </div>
    </li>
  );

  const renderTranscript = () => {
    const completedItems = items.filter(item => item.completed);
    const openItem = items.find(item => !item.completed);
    // The open item's text is the stable prefix the server already streamed as deltas
    const stable = openItem?.text ?? '';
    const showHypothesis = hypothesis.text.length > 0;
    const isEmpty = completedItems.length === 0 && !showHypothesis && !stable;

    if (isEmpty) {
      return (
        <div className="h-full flex flex-col items-center justify-center text-center text-gray-500 p-8">
          <svg className="w-12 h-12 mb-3 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
          </svg>
          <p className="font-medium">
            {phase === 'live' && serverStatus === 'listening'
              ? t('liveTranscription.speakNow')
              : t('liveTranscription.emptyTitle')}
          </p>
          <p className="text-sm mt-1">{t('liveTranscription.emptyHint')}</p>
        </div>
      );
    }

    return (
      <ul className="divide-y divide-gray-100">
        {completedItems.map(item => renderSegmentRow(item.id, item.segment, item.text, false))}
        {(showHypothesis || stable) && renderSegmentRow(
          'hypothesis',
          hypothesis.segments[0] ?? null,
          showHypothesis && hypothesis.text.startsWith(stable) ? (
            <>
              <span className="text-gray-900">{stable}</span>
              <span className="text-gray-400 italic">{hypothesis.text.slice(stable.length)}</span>
            </>
          ) : (
            <span className="text-gray-400 italic">{showHypothesis ? hypothesis.text : stable}</span>
          ),
          true,
        )}
      </ul>
    );
  };

  const renderControls = () => (
    <div className="space-y-5">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className={`w-3 h-3 rounded-full ${phase === 'idle' ? 'bg-gray-300' : statusDotClass()}`} />
          <span className="text-sm font-medium text-gray-700">
            {phase === 'idle' ? t('liveTranscription.status.idle') : statusLabel()}
          </span>
        </div>
        <span className="text-2xl font-mono text-gray-900">{formatElapsed(elapsed)}</span>
      </div>

      <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
        <div
          className="bg-emerald-500 h-2 rounded-full transition-all duration-100"
          style={{ width: `${Math.min(1, level * 4) * 100}%` }}
        />
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">{t('transcription.contextInfo')}</label>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={3}
          maxLength={2000}
          disabled={isActive}
          placeholder={t('transcription.contextInfoPlaceholder')}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-50"
        />
        <p className="text-xs text-gray-500 mt-1">{t('transcription.contextInfoHint')}</p>
      </div>

      <div>
        <label className="block text-sm font-medium text-gray-700 mb-2">{t('transcription.modelDtype')}</label>
        <select
          value={model}
          onChange={(e) => setModel(e.target.value)}
          disabled={isActive}
          className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm disabled:bg-gray-50"
        >
          {MODELS.map(option => (
            <option key={option.name} value={option.name}>{option.dtype}</option>
          ))}
        </select>
        <p className="text-xs text-gray-500 mt-1">{t('transcription.modelDtypeHint')}</p>
      </div>

      <details className="border border-gray-200 rounded-lg">
        <summary className="px-4 py-3 cursor-pointer text-sm font-medium text-gray-700">
          {t('transcription.advancedSettings')}
        </summary>
        <div className="px-4 pb-4 space-y-4">
          <div>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={offloadingEnabled}
                onChange={(e) => setOffloadingEnabled(e.target.checked)}
                disabled={isActive}
                className="w-4 h-4 text-blue-500 rounded focus:ring-2 focus:ring-blue-500"
              />
              <span className="text-sm font-medium text-gray-700">{t('generation.enableOffloading')}</span>
            </label>
            {offloadingEnabled && (
              <select
                value={offloadingPreset}
                onChange={(e) => setOffloadingPreset(e.target.value as OffloadingPreset)}
                disabled={isActive}
                className="mt-3 w-full px-3 py-2 border border-gray-300 rounded-lg text-sm disabled:bg-gray-50"
              >
                {(Object.keys(PRESET_GPU_LAYERS) as OffloadingPreset[]).map(preset => (
                  <option key={preset} value={preset}>
                    {t(`generation.${preset}`)} ({t('generation.gpuLayers')}: {PRESET_GPU_LAYERS[preset]})
                  </option>
                ))}
              </select>
            )}
          </div>

          <div className="pt-3 border-t border-gray-200">
            <label className="block text-xs text-gray-600 mb-1">{t('liveTranscription.apiKey')}</label>
            <input
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              disabled={isActive}
              autoComplete="off"
              placeholder={t('liveTranscription.apiKeyPlaceholder')}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg text-sm disabled:bg-gray-50"
            />
            <p className="text-xs text-gray-500 mt-1">{t('liveTranscription.apiKeyHint')}</p>
          </div>
        </div>
      </details>

      {error && (
        <div className="p-3 bg-red-50 border border-red-200 rounded-lg">
          <p className="text-sm text-red-800">{error}</p>
          {phase === 'error' && requestId && (
            <Link
              href={`${recordPath}?request_id=${encodeURIComponent(requestId)}`}
              className="text-sm text-red-700 underline mt-1 inline-block"
            >
              {t('liveTranscription.viewRecord')}
            </Link>
          )}
        </div>
      )}

      {phase === 'done' && (
        <div className="p-3 bg-green-50 border border-green-200 rounded-lg">
          <p className="text-sm text-green-800">
            {requestId ? t('liveTranscription.saved') : t('liveTranscription.nothingRecorded')}
          </p>
          {requestId && (
            <Link
              href={`${recordPath}?request_id=${encodeURIComponent(requestId)}`}
              className="text-sm text-green-700 underline mt-1 inline-block"
            >
              {t('liveTranscription.viewRecord')}
            </Link>
          )}
        </div>
      )}

      {isActive ? (
        <button
          onClick={stopSession}
          disabled={phase === 'stopping'}
          className="w-full px-4 py-3 bg-red-500 text-white font-medium rounded-lg hover:bg-red-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center gap-2"
        >
          {phase === 'stopping' ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white" />
              {t('liveTranscription.stopping')}
            </>
          ) : (
            <>
              <span className="w-3 h-3 bg-white rounded-sm" />
              {t('liveTranscription.stop')}
            </>
          )}
        </button>
      ) : (
        <button
          onClick={startSession}
          className="w-full px-4 py-3 bg-blue-500 text-white font-medium rounded-lg hover:bg-blue-600 transition-colors flex items-center justify-center gap-2"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
          </svg>
          {phase === 'idle' ? t('liveTranscription.start') : t('liveTranscription.startNew')}
        </button>
      )}
    </div>
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
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-lg font-semibold text-gray-900">
                {t('liveTranscription.transcript')}
                <span className="ml-2 text-sm font-normal text-gray-500">
                  ({t('transcription.segmentCount', { count: items.filter(item => item.completed).length })})
                </span>
              </h2>
              <button
                onClick={copyTranscript}
                disabled={!transcriptText()}
                className="px-3 py-1.5 text-sm border border-gray-300 rounded-lg hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {t('transcription.copyText')}
              </button>
            </div>
            <div className="flex-1 overflow-y-auto border border-gray-200 rounded-lg">
              {renderTranscript()}
              <div ref={transcriptEndRef} />
            </div>
          </div>

          <div className="flex flex-col gap-6 overflow-y-auto">
            <div className="bg-white rounded-lg shadow-sm p-6">
              {renderControls()}
            </div>

            <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
              <h3 className="text-sm font-semibold text-blue-900 mb-2">{t('transcription.howItWorks')}</h3>
              <ul className="text-sm text-blue-800 space-y-1">
                <li>{t('liveTranscription.step1')}</li>
                <li>{t('liveTranscription.step2')}</li>
                <li>{t('liveTranscription.step3')}</li>
                <li>{t('liveTranscription.step4')}</li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
