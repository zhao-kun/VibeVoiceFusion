/**
 * Transcription (ASR) types for the Transcription API
 */

import { InferencePhase, ModelDtype, OffloadingConfig } from './generation';

/**
 * One speaker-labelled segment produced by the ASR model
 */
export interface TranscriptionSegment {
  start_time: number | string | null;
  end_time: number | string | null;
  speaker_id: number | string | null;
  text: string;
}

/**
 * Timing and token statistics of a transcription
 */
export interface TranscriptionDetails {
  preprocessing_duration?: number | null;
  model_load_duration?: number | null;
  encode_time?: number | null;
  prefill_time?: number | null;
  decode_time?: number | null;
  prompt_tokens?: number | null;
  speech_tokens?: number | null;
  offloading_config?: OffloadingConfig | Record<string, never> | null;
}

/**
 * Transcription metadata from backend
 */
export interface Transcription {
  request_id: string;
  audio_file: string;
  original_filename: string;
  status: InferencePhase;
  model_dtype: ModelDtype;
  max_new_tokens: number;
  temperature: number;
  top_p: number;
  repetition_penalty: number;
  seeds: number;
  created_at: string;
  updated_at: string;
  project_id: string | null;
  context_info: string | null;
  offloading: OffloadingConfig | null;
  audio_duration: number | null;
  generated_tokens: number;
  reach_max_new_tokens: boolean;
  raw_text: string | null;
  segments: TranscriptionSegment[];
  text_preview: string | null;
  percentage: number | null;
  details: TranscriptionDetails | null;
  error_message: string | null;
  completed_at: string | null;
}

/**
 * Request parameters to start a transcription
 */
export interface StartTranscriptionRequest {
  audio_file: File;
  context_info?: string;
  model_dtype?: ModelDtype;
  max_new_tokens?: number;
  temperature?: number;
  top_p?: number;
  repetition_penalty?: number;
  seeds?: number;
  offloading?: OffloadingConfig;
}

/**
 * Response when starting a transcription
 */
export interface StartTranscriptionResponse {
  message: string;
  request_id: string;
  project_id: string | null;
  status: InferencePhase;
}

/**
 * Response from the current transcription endpoints
 */
export interface CurrentTranscriptionResponse {
  message: string;
  transcription: Transcription | null;
}

/**
 * Summary entry in transcription history
 */
export interface TranscriptionSummary {
  request_id: string;
  project_id: string | null;
  status: InferencePhase;
  original_filename: string;
  audio_duration: number | null;
  text_preview: string | null;
  segment_count: number;
  model_dtype: ModelDtype;
  created_at: string;
  completed_at: string | null;
}

/**
 * Response from the transcription history endpoints
 */
export interface TranscriptionHistoryResponse {
  transcriptions: TranscriptionSummary[];
  count: number;
  total: number;
}

/**
 * Response from the batch delete endpoints
 */
export interface BatchDeleteTranscriptionsResponse {
  message: string;
  deleted_count: number;
  failed_count: number;
  deleted_ids: string[];
  failed_ids: string[];
}

/**
 * Transcript download formats
 */
export type TranscriptFormat = 'json' | 'txt';

/**
 * Whether a transcription is still being processed
 */
export function isTranscriptionActive(status: InferencePhase | string): boolean {
  return status === InferencePhase.PENDING
    || status === InferencePhase.PREPROCESSING
    || status === InferencePhase.INFERENCING;
}

/**
 * Parse a segment timestamp into seconds
 */
export function segmentSeconds(value: number | string | null | undefined): number | null {
  if (value === null || value === undefined || value === '') return null;
  const parsed = typeof value === 'number' ? value : parseFloat(value);
  return Number.isFinite(parsed) ? parsed : null;
}

/**
 * Format seconds as mm:ss.ss (or h:mm:ss.ss)
 */
export function formatTimestamp(seconds: number | null): string {
  if (seconds === null) return '--:--';
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = (seconds % 60).toFixed(2).padStart(5, '0');
  const mm = String(minutes).padStart(2, '0');
  return hours > 0 ? `${hours}:${mm}:${secs}` : `${mm}:${secs}`;
}
