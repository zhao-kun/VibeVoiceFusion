/**
 * Backend API client for VibeVoice
 */

import type {
  CreateGenerationRequest,
  CreateGenerationResponse,
  CurrentGenerationResponse,
  ListGenerationsResponse,
  GetGenerationResponse
} from '@/types/generation';

import type {
  CreateTrainingRequest,
  CreateTrainingResponse,
  CurrentTrainingResponse,
  ListTrainingStatesResponse,
  GetTrainingStateResponse,
  DeleteTrainingResponse,
  BatchDeleteTrainingResponse,
  GetTrainingMetricsResponse
} from '@/types/training';

import type { CurrentTaskResponse } from '@/types/task';
import type {
  PresetVoice,
  PresetLanguage,
  ListPresetsResponse,
  DeletePresetResponse,
  BatchDeletePresetsResponse
} from '@/types/preset';

import type {
  QuickGenerate,
  StartQuickGenerateResponse,
  CurrentQuickGenerateResponse,
  QuickGenerateHistoryResponse
} from '@/types/quickGenerate';
import type {
  Transcription,
  StartTranscriptionRequest,
  StartTranscriptionResponse,
  CurrentTranscriptionResponse,
  TranscriptionHistoryResponse,
  BatchDeleteTranscriptionsResponse,
  TranscriptFormat,
} from '@/types/transcription';

import type { OffloadingConfig } from '@/types/generation';

// API base URL configuration
// Development: Full URL to backend server (different origin)
// Production: Relative path (same origin, backend serves frontend)
const API_BASE_URL = process.env.NODE_ENV === 'development'
  ? (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:9527/api/v1')
  : '/api/v1';

export interface Project {
  id: string;
  name: string;
  description: string;
  created_at: string;
  updated_at: string;
}

export interface Speaker {
  speaker_id: string;
  description: string;
  voice_filename: string;
  created_at: string;
  updated_at: string;
}

export type SessionMode = 'dialogue' | 'narration';

export interface DialogSession {
  session_id: string;
  name: string;
  description: string;
  text_filename: string;
  created_at: string;
  updated_at: string;
  mode: SessionMode;
  narrator_speaker_id: string | null;
}

export interface Dataset {
  id: string;
  name: string;
  description: string;
  item_count: number;
  created_at: string;
  updated_at: string;
}

export interface LoRAFile {
  display_name: string;
  full_path: string;
  lora_name: string;
  filename: string;
}

export interface DatasetItem {
  text: string;
  audio: string;
  voice_prompts: string[];
}

class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  private async fetch<T>(
    endpoint: string,
    options?: RequestInit
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;

    // Get current language from localStorage
    const locale = typeof window !== 'undefined'
      ? localStorage.getItem('vibevoice-locale') || 'en'
      : 'en';

    try {
      const response = await fetch(url, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          'X-Language': locale,
          ...options?.headers,
        },
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({
          error: 'Unknown error',
          message: response.statusText
        }));
        throw new Error(error.message || error.error || response.statusText);
      }

      return await response.json();
    } catch (error) {
      if (error instanceof Error) {
        throw error;
      }
      throw new Error('Network error');
    }
  }

  // ============ Projects API ============

  async listProjects(): Promise<{ projects: Project[]; count: number }> {
    return this.fetch('/projects');
  }

  async getProject(projectId: string): Promise<Project> {
    return this.fetch(`/projects/${encodeURIComponent(projectId)}`);
  }

  async createProject(data: {
    name: string;
    description?: string;
  }): Promise<Project> {
    return this.fetch('/projects', {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async updateProject(
    projectId: string,
    data: { name?: string; description?: string }
  ): Promise<Project> {
    return this.fetch(`/projects/${encodeURIComponent(projectId)}`, {
      method: 'PUT',
      body: JSON.stringify(data),
    });
  }

  async deleteProject(projectId: string): Promise<{ message: string; project_id: string }> {
    return this.fetch(`/projects/${encodeURIComponent(projectId)}`, {
      method: 'DELETE',
    });
  }

  // ============ Speakers API ============

  async listSpeakers(projectId: string): Promise<{ speakers: Speaker[]; count: number }> {
    return this.fetch(`/projects/${encodeURIComponent(projectId)}/speakers`);
  }

  async getSpeaker(projectId: string, speakerId: string): Promise<Speaker> {
    return this.fetch(
      `/projects/${encodeURIComponent(projectId)}/speakers/${encodeURIComponent(speakerId)}`
    );
  }

  async createSpeaker(
    projectId: string,
    data: {
      description?: string;
      voice_file: File;
    }
  ): Promise<Speaker> {
    const formData = new FormData();
    if (data.description) {
      formData.append('description', data.description);
    }
    formData.append('voice_file', data.voice_file);

    const url = `${this.baseUrl}/projects/${encodeURIComponent(projectId)}/speakers`;
    const response = await fetch(url, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        error: 'Unknown error',
        message: response.statusText
      }));
      throw new Error(error.message || error.error || response.statusText);
    }

    return await response.json();
  }

  async updateSpeaker(
    projectId: string,
    speakerId: string,
    data: { description?: string }
  ): Promise<Speaker> {
    return this.fetch(
      `/projects/${encodeURIComponent(projectId)}/speakers/${encodeURIComponent(speakerId)}`,
      {
        method: 'PUT',
        body: JSON.stringify(data),
      }
    );
  }

  async deleteSpeaker(
    projectId: string,
    speakerId: string
  ): Promise<{ message: string; speaker_id: string }> {
    return this.fetch(
      `/projects/${encodeURIComponent(projectId)}/speakers/${encodeURIComponent(speakerId)}`,
      {
        method: 'DELETE',
      }
    );
  }

  async updateVoiceFile(
    projectId: string,
    speakerId: string,
    voiceFile: File
  ): Promise<Speaker> {
    const formData = new FormData();
    formData.append('voice_file', voiceFile);

    const url = `${this.baseUrl}/projects/${encodeURIComponent(projectId)}/speakers/${encodeURIComponent(speakerId)}/voice`;
    const response = await fetch(url, {
      method: 'PUT',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        error: 'Unknown error',
        message: response.statusText
      }));
      throw new Error(error.message || error.error || response.statusText);
    }

    return await response.json();
  }

  async trimVoiceFile(
    projectId: string,
    speakerId: string,
    startTime: number,
    endTime: number
  ): Promise<Speaker> {
    return this.fetch(
      `/projects/${encodeURIComponent(projectId)}/speakers/${encodeURIComponent(speakerId)}/voice/trim`,
      {
        method: 'POST',
        body: JSON.stringify({ start_time: startTime, end_time: endTime }),
      }
    );
  }

  getVoiceFileUrl(projectId: string, speakerId: string): string {
    return `${this.baseUrl}/projects/${encodeURIComponent(projectId)}/speakers/${encodeURIComponent(speakerId)}/voice`;
  }

  async createSpeakerFromPreset(
    projectId: string,
    data: {
      preset_filename: string;
      description?: string;
    }
  ): Promise<Speaker> {
    return this.fetch(
      `/projects/${encodeURIComponent(projectId)}/speakers/from-preset`,
      {
        method: 'POST',
        body: JSON.stringify(data),
      }
    );
  }

  // ============ Preset Voices API ============

  async listPresetVoices(options?: {
    language?: string;
    gender?: "man" | "woman";
    has_bgm?: boolean;
    offset?: number;
    limit?: number;
  }): Promise<ListPresetsResponse> {
    const params = new URLSearchParams();
    if (options?.language) params.append('language', options.language);
    if (options?.gender) params.append('gender', options.gender);
    if (options?.has_bgm !== undefined) {
      params.append('has_bgm', options.has_bgm.toString());
    }
    if (options?.offset !== undefined) {
      params.append('offset', options.offset.toString());
    }
    if (options?.limit !== undefined) {
      params.append('limit', options.limit.toString());
    }

    const queryString = params.toString();
    return this.fetch(`/preset-voices${queryString ? '?' + queryString : ''}`);
  }

  async getPresetVoice(filename: string): Promise<PresetVoice> {
    return this.fetch(`/preset-voices/${encodeURIComponent(filename)}`);
  }

  async createPresetVoice(data: {
    name: string;
    language: string;
    gender: "man" | "woman";
    has_bgm: boolean;
    voice_file: File;
  }): Promise<PresetVoice> {
    const formData = new FormData();
    formData.append('name', data.name);
    formData.append('language', data.language);
    formData.append('gender', data.gender);
    formData.append('has_bgm', data.has_bgm.toString());
    formData.append('voice_file', data.voice_file);

    const url = `${this.baseUrl}/preset-voices`;
    const locale = typeof window !== 'undefined'
      ? localStorage.getItem('vibevoice-locale') || 'en'
      : 'en';

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'X-Language': locale,
      },
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        error: 'Unknown error',
        message: response.statusText
      }));
      throw new Error(error.message || error.error || response.statusText);
    }

    return await response.json();
  }

  async deletePresetVoice(filename: string): Promise<DeletePresetResponse> {
    return this.fetch(`/preset-voices/${encodeURIComponent(filename)}`, {
      method: 'DELETE',
    });
  }

  async batchDeletePresetVoices(
    filenames: string[]
  ): Promise<BatchDeletePresetsResponse> {
    return this.fetch('/preset-voices/batch-delete', {
      method: 'POST',
      body: JSON.stringify({ filenames }),
    });
  }

  async listPresetLanguages(): Promise<{ languages: PresetLanguage[] }> {
    return this.fetch('/preset-voices/languages');
  }

  getPresetPreviewUrl(filename: string): string {
    return `${this.baseUrl}/preset-voices/${encodeURIComponent(filename)}/preview`;
  }

  // ============ Dialog Sessions API ============

  async listSessions(projectId: string): Promise<{ sessions: DialogSession[]; count: number }> {
    return this.fetch(`/projects/${encodeURIComponent(projectId)}/sessions`);
  }

  async getSession(projectId: string, sessionId: string): Promise<DialogSession> {
    return this.fetch(
      `/projects/${encodeURIComponent(projectId)}/sessions/${encodeURIComponent(sessionId)}`
    );
  }

  async createSession(
    projectId: string,
    data: {
      name: string;
      description?: string;
      dialog_text: string;
      mode?: SessionMode;
      narrator_speaker_id?: string;
    }
  ): Promise<DialogSession> {
    return this.fetch(`/projects/${encodeURIComponent(projectId)}/sessions`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async updateSession(
    projectId: string,
    sessionId: string,
    data: {
      name?: string;
      description?: string;
      dialog_text?: string;
      mode?: SessionMode;
      narrator_speaker_id?: string;
    }
  ): Promise<DialogSession> {
    return this.fetch(
      `/projects/${encodeURIComponent(projectId)}/sessions/${encodeURIComponent(sessionId)}`,
      {
        method: 'PUT',
        body: JSON.stringify(data),
      }
    );
  }

  async deleteSession(
    projectId: string,
    sessionId: string
  ): Promise<{ message: string; session_id: string }> {
    return this.fetch(
      `/projects/${encodeURIComponent(projectId)}/sessions/${encodeURIComponent(sessionId)}`,
      {
        method: 'DELETE',
      }
    );
  }

  async getSessionText(
    projectId: string,
    sessionId: string
  ): Promise<{ session_id: string; dialog_text: string }> {
    return this.fetch(
      `/projects/${encodeURIComponent(projectId)}/sessions/${encodeURIComponent(sessionId)}/text`
    );
  }

  getSessionDownloadUrl(projectId: string, sessionId: string): string {
    return `${this.baseUrl}/projects/${encodeURIComponent(projectId)}/sessions/${encodeURIComponent(sessionId)}/download`;
  }

  // ============ Tasks API (Unified) ============

  async getCurrentTask(): Promise<CurrentTaskResponse> {
    return this.fetch('/tasks/current');
  }

  // ============ Generations API ============

  async createGeneration(
    projectId: string,
    data: CreateGenerationRequest
  ): Promise<CreateGenerationResponse> {
    return this.fetch(`/projects/${encodeURIComponent(projectId)}/generations`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async getCurrentGeneration(): Promise<CurrentGenerationResponse> {
    return this.fetch('/projects/generations/current');
  }

  async getCurrentGenerationForProject(projectId: string): Promise<CurrentGenerationResponse> {
    return this.fetch(`/projects/${encodeURIComponent(projectId)}/generations/current`);
  }

  async listGenerations(projectId: string): Promise<ListGenerationsResponse> {
    return this.fetch(`/projects/${encodeURIComponent(projectId)}/generations`);
  }

  async getGeneration(projectId: string, requestId: string): Promise<GetGenerationResponse> {
    return this.fetch(`/projects/${encodeURIComponent(projectId)}/generations/${encodeURIComponent(requestId)}`);
  }

  async deleteGeneration(
    projectId: string,
    requestId: string
  ): Promise<{ message: string; request_id: string }> {
    return this.fetch(
      `/projects/${encodeURIComponent(projectId)}/generations/${encodeURIComponent(requestId)}`,
      {
        method: 'DELETE',
      }
    );
  }

  async batchDeleteGenerations(
    projectId: string,
    requestIds: string[]
  ): Promise<{
    message: string;
    deleted_count: number;
    failed_count: number;
    deleted_ids: string[];
    failed_ids: string[];
  }> {
    return this.fetch(
      `/projects/${encodeURIComponent(projectId)}/generations/batch-delete`,
      {
        method: 'POST',
        body: JSON.stringify({ request_ids: requestIds }),
      }
    );
  }

  getGenerationDownloadUrl(projectId: string, requestId: string): string {
    return `${this.baseUrl}/projects/${encodeURIComponent(projectId)}/generations/${encodeURIComponent(requestId)}/download`;
  }

  getGenerationItemDownloadUrl(projectId: string, requestId: string, itemIndex: number): string {
    return `${this.baseUrl}/projects/${encodeURIComponent(projectId)}/generations/${encodeURIComponent(requestId)}/items/${itemIndex}/download`;
  }

  // ============ Datasets API ============

  async listDatasets(projectId: string): Promise<{ datasets: Dataset[]; count: number }> {
    return this.fetch(`/projects/${encodeURIComponent(projectId)}/datasets`);
  }

  async getDataset(projectId: string, datasetId: string): Promise<Dataset> {
    return this.fetch(
      `/projects/${encodeURIComponent(projectId)}/datasets/${encodeURIComponent(datasetId)}`
    );
  }

  async createDataset(
    projectId: string,
    data: {
      name: string;
      description?: string;
    }
  ): Promise<Dataset> {
    return this.fetch(`/projects/${encodeURIComponent(projectId)}/datasets`, {
      method: 'POST',
      body: JSON.stringify(data),
    });
  }

  async updateDataset(
    projectId: string,
    datasetId: string,
    data: { name?: string; description?: string }
  ): Promise<Dataset> {
    return this.fetch(
      `/projects/${encodeURIComponent(projectId)}/datasets/${encodeURIComponent(datasetId)}`,
      {
        method: 'PUT',
        body: JSON.stringify(data),
      }
    );
  }

  async deleteDataset(
    projectId: string,
    datasetId: string
  ): Promise<{ message: string; dataset_id: string }> {
    return this.fetch(
      `/projects/${encodeURIComponent(projectId)}/datasets/${encodeURIComponent(datasetId)}`,
      {
        method: 'DELETE',
      }
    );
  }

  getDatasetExportUrl(projectId: string, datasetId: string): string {
    return `${this.baseUrl}/projects/${encodeURIComponent(projectId)}/datasets/${encodeURIComponent(datasetId)}/export`;
  }

  async importToExistingDataset(
    projectId: string,
    datasetId: string,
    datasetFile: File
  ): Promise<Dataset> {
    const formData = new FormData();
    formData.append('dataset_file', datasetFile);

    const url = `${this.baseUrl}/projects/${encodeURIComponent(projectId)}/datasets/${encodeURIComponent(datasetId)}/import`;
    const locale = typeof window !== 'undefined'
      ? localStorage.getItem('vibevoice-locale') || 'en'
      : 'en';

    const response = await fetch(url, {
      method: 'PUT',
      headers: {
        'X-Language': locale,
      },
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        error: 'Unknown error',
        message: response.statusText
      }));
      throw new Error(error.message || error.error || response.statusText);
    }

    return await response.json();
  }

  async importDataset(
    projectId: string,
    data: {
      dataset_file: File;
      name?: string;
    }
  ): Promise<Dataset> {
    const formData = new FormData();
    formData.append('dataset_file', data.dataset_file);
    if (data.name) {
      formData.append('name', data.name);
    }

    const url = `${this.baseUrl}/projects/${encodeURIComponent(projectId)}/datasets/import`;
    const locale = typeof window !== 'undefined'
      ? localStorage.getItem('vibevoice-locale') || 'en'
      : 'en';

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'X-Language': locale,
      },
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        error: 'Unknown error',
        message: response.statusText
      }));
      throw new Error(error.message || error.error || response.statusText);
    }

    return await response.json();
  }

  // ============ Dataset Items API ============

  async listDatasetItems(
    projectId: string,
    datasetId: string,
    options?: {
      offset?: number;
      limit?: number;
    }
  ): Promise<{
    items: DatasetItem[];
    count: number;
    total: number;
    offset: number;
    limit: number | null;
  }> {
    const params = new URLSearchParams();
    if (options?.offset !== undefined) {
      params.append('offset', options.offset.toString());
    }
    if (options?.limit !== undefined) {
      params.append('limit', options.limit.toString());
    }

    const queryString = params.toString();
    const endpoint = `/projects/${encodeURIComponent(projectId)}/datasets/${encodeURIComponent(datasetId)}/items${queryString ? `?${queryString}` : ''}`;

    return this.fetch(endpoint);
  }

  async createDatasetItem(
    projectId: string,
    datasetId: string,
    data: {
      text: string;
      audio_file: File;
      voice_prompt_files: File[];
    }
  ): Promise<DatasetItem> {
    const formData = new FormData();
    formData.append('text', data.text);
    formData.append('audio_file', data.audio_file);
    data.voice_prompt_files.forEach((file) => {
      formData.append('voice_prompt_files', file);
    });

    const url = `${this.baseUrl}/projects/${encodeURIComponent(projectId)}/datasets/${encodeURIComponent(datasetId)}/items`;
    const locale = typeof window !== 'undefined'
      ? localStorage.getItem('vibevoice-locale') || 'en'
      : 'en';

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'X-Language': locale,
      },
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        error: 'Unknown error',
        message: response.statusText
      }));
      throw new Error(error.message || error.error || response.statusText);
    }

    return await response.json();
  }

  async updateDatasetItem(
    projectId: string,
    datasetId: string,
    itemIndex: number,
    data: {
      text?: string;
      audio_file?: File;
      voice_prompt_files?: File[];
    }
  ): Promise<DatasetItem> {
    const formData = new FormData();
    if (data.text !== undefined) {
      formData.append('text', data.text);
    }
    if (data.audio_file) {
      formData.append('audio_file', data.audio_file);
    }
    if (data.voice_prompt_files) {
      data.voice_prompt_files.forEach((file) => {
        formData.append('voice_prompt_files', file);
      });
    }

    const url = `${this.baseUrl}/projects/${encodeURIComponent(projectId)}/datasets/${encodeURIComponent(datasetId)}/items/${itemIndex}`;
    const locale = typeof window !== 'undefined'
      ? localStorage.getItem('vibevoice-locale') || 'en'
      : 'en';

    const response = await fetch(url, {
      method: 'PUT',
      headers: {
        'X-Language': locale,
      },
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        error: 'Unknown error',
        message: response.statusText
      }));
      throw new Error(error.message || error.error || response.statusText);
    }

    return await response.json();
  }

  async deleteDatasetItem(
    projectId: string,
    datasetId: string,
    itemIndex: number
  ): Promise<{ message: string; item_index: number }> {
    return this.fetch(
      `/projects/${encodeURIComponent(projectId)}/datasets/${encodeURIComponent(datasetId)}/items/${itemIndex}`,
      {
        method: 'DELETE',
      }
    );
  }

  // ============ Training API ============

  async createTrainingJob(
    projectId: string,
    request: CreateTrainingRequest
  ): Promise<CreateTrainingResponse> {
    return this.fetch(`/projects/${encodeURIComponent(projectId)}/training`, {
      method: 'POST',
      body: JSON.stringify(request),
    });
  }

  async listTrainingStates(
    projectId: string
  ): Promise<ListTrainingStatesResponse> {
    return this.fetch(`/projects/${encodeURIComponent(projectId)}/training`);
  }

  async getCurrentTrainingState(
    projectId: string
  ): Promise<CurrentTrainingResponse> {
    return this.fetch(`/projects/${encodeURIComponent(projectId)}/training/current`);
  }

  async getTrainingState(
    projectId: string,
    jobId: string
  ): Promise<GetTrainingStateResponse> {
    return this.fetch(
      `/projects/${encodeURIComponent(projectId)}/training/${encodeURIComponent(jobId)}`
    );
  }

  async deleteTrainingJob(
    projectId: string,
    jobId: string
  ): Promise<DeleteTrainingResponse> {
    return this.fetch(
      `/projects/${encodeURIComponent(projectId)}/training/${encodeURIComponent(jobId)}`,
      {
        method: 'DELETE',
      }
    );
  }

  async batchDeleteTrainingJobs(
    projectId: string,
    jobIds: string[]
  ): Promise<BatchDeleteTrainingResponse> {
    return this.fetch(
      `/projects/${encodeURIComponent(projectId)}/training/batch-delete`,
      {
        method: 'POST',
        body: JSON.stringify({ job_ids: jobIds }),
      }
    );
  }

  /**
   * Get training metrics from TensorBoard logs
   * Returns null if metrics are not yet available (404 during prepare phase)
   */
  async getTrainingMetrics(
    projectId: string,
    jobId: string,
    options?: {
      maxPoints?: number;
      metrics?: 'all' | 'loss' | 'learning_rate' | 'timing' | string;
    }
  ): Promise<GetTrainingMetricsResponse | null> {
    const params = new URLSearchParams();
    if (options?.maxPoints) {
      params.append('max_points', options.maxPoints.toString());
    }
    if (options?.metrics) {
      params.append('metrics', options.metrics);
    }

    const queryString = params.toString();
    const endpoint = `/projects/${encodeURIComponent(projectId)}/training/${encodeURIComponent(jobId)}/metrics${queryString ? '?' + queryString : ''}`;
    const url = `${this.baseUrl}${endpoint}`;

    // Get current language from localStorage
    const locale = typeof window !== 'undefined'
      ? localStorage.getItem('vibevoice-locale') || 'en'
      : 'en';

    const response = await fetch(url, {
      headers: {
        'Content-Type': 'application/json',
        'X-Language': locale,
      },
    });

    // Return null for 404 - metrics not yet available during prepare phase
    if (response.status === 404) {
      return null;
    }

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        error: 'Unknown error',
        message: response.statusText
      }));
      throw new Error(error.message || error.error || response.statusText);
    }

    return await response.json();
  }

  /**
   * Download a LoRA file from a completed training job
   */
  downloadLoRAFile(projectId: string, jobId: string, filename: string): void {
    const url = `${this.baseUrl}/projects/${encodeURIComponent(projectId)}/training/${encodeURIComponent(jobId)}/lora/${encodeURIComponent(filename)}`;

    // Create a temporary anchor element to trigger download
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  /**
   * List available LoRA files for a project
   */
  async listLoRAFiles(projectId: string): Promise<{ lora_files: LoRAFile[]; count: number }> {
    return this.fetch(`/projects/${encodeURIComponent(projectId)}/training/lora-files`);
  }

  // ============ Quick Generate API ============

  /**
   * Start a quick generation task
   */
  async startQuickGeneration(data: {
    voice_files: File[];
    text: string;
    seeds?: number;
    batch_size?: number;
    cfg_scale?: number;
    model_dtype?: string;
    attn_implementation?: string;
    offloading?: OffloadingConfig;
  }): Promise<StartQuickGenerateResponse> {
    const formData = new FormData();

    // Append voice files with indexed keys (voice_file_0, voice_file_1, etc.)
    data.voice_files.forEach((file, index) => {
      formData.append(`voice_file_${index}`, file);
    });

    formData.append('text', data.text);

    if (data.seeds !== undefined) {
      formData.append('seeds', data.seeds.toString());
    }
    if (data.batch_size !== undefined) {
      formData.append('batch_size', data.batch_size.toString());
    }
    if (data.cfg_scale !== undefined) {
      formData.append('cfg_scale', data.cfg_scale.toString());
    }
    if (data.model_dtype) {
      formData.append('model_dtype', data.model_dtype);
    }
    if (data.attn_implementation) {
      formData.append('attn_implementation', data.attn_implementation);
    }
    if (data.offloading) {
      formData.append('offloading', JSON.stringify(data.offloading));
    }

    const url = `${this.baseUrl}/quick-generate`;
    const locale = typeof window !== 'undefined'
      ? localStorage.getItem('vibevoice-locale') || 'en'
      : 'en';

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'X-Language': locale,
      },
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        error: 'Unknown error',
        message: response.statusText
      }));
      throw new Error(error.message || error.error || response.statusText);
    }

    return await response.json();
  }

  /**
   * Get status of a quick generation request
   */
  async getQuickGeneration(requestId: string): Promise<QuickGenerate> {
    return this.fetch(`/quick-generate/${encodeURIComponent(requestId)}`);
  }

  /**
   * Get current running quick generation task
   */
  async getCurrentQuickGeneration(): Promise<CurrentQuickGenerateResponse> {
    return this.fetch('/quick-generate/current');
  }

  /**
   * List quick generation history
   */
  async listQuickGenerationHistory(options?: {
    limit?: number;
    offset?: number;
  }): Promise<QuickGenerateHistoryResponse> {
    const params = new URLSearchParams();
    if (options?.limit !== undefined) {
      params.append('limit', options.limit.toString());
    }
    if (options?.offset !== undefined) {
      params.append('offset', options.offset.toString());
    }

    const queryString = params.toString();
    return this.fetch(`/quick-generate/history${queryString ? '?' + queryString : ''}`);
  }

  /**
   * Delete a quick generation
   */
  async deleteQuickGeneration(requestId: string): Promise<{ message: string; request_id: string }> {
    return this.fetch(`/quick-generate/${encodeURIComponent(requestId)}`, {
      method: 'DELETE',
    });
  }

  /**
   * Get download URL for quick generation audio
   */
  getQuickGenerationDownloadUrl(requestId: string): string {
    return `${this.baseUrl}/quick-generate/${encodeURIComponent(requestId)}/download`;
  }

  /**
   * Get download URL for quick generation item (multi-generation)
   */
  getQuickGenerationItemDownloadUrl(requestId: string, itemIndex: number): string {
    return `${this.baseUrl}/quick-generate/${encodeURIComponent(requestId)}/items/${itemIndex}/download`;
  }

  /**
   * Get voice preview URL for quick generation
   */
  getQuickGenerationVoicePreviewUrl(requestId: string): string {
    return `${this.baseUrl}/quick-generate/${encodeURIComponent(requestId)}/voice/preview`;
  }

  /**
   * Get voice preview URL for quick generation by index
   */
  getQuickGenerationVoicePreviewByIndexUrl(requestId: string, voiceIndex: number): string {
    return `${this.baseUrl}/quick-generate/${encodeURIComponent(requestId)}/voice/${voiceIndex}/preview`;
  }

  // ============ Transcription API ============
  // A null projectId targets standalone transcriptions (/transcriptions), otherwise /projects/{id}/transcriptions

  private transcriptionBase(projectId: string | null): string {
    return projectId ? `/projects/${encodeURIComponent(projectId)}/transcriptions` : '/transcriptions';
  }

  /**
   * Start a transcription task
   */
  async startTranscription(
    projectId: string | null,
    data: StartTranscriptionRequest
  ): Promise<StartTranscriptionResponse> {
    const formData = new FormData();
    formData.append('audio_file', data.audio_file);

    if (data.context_info) {
      formData.append('context_info', data.context_info);
    }
    if (data.model_dtype) {
      formData.append('model_dtype', data.model_dtype);
    }
    if (data.max_new_tokens !== undefined) {
      formData.append('max_new_tokens', data.max_new_tokens.toString());
    }
    if (data.temperature !== undefined) {
      formData.append('temperature', data.temperature.toString());
    }
    if (data.top_p !== undefined) {
      formData.append('top_p', data.top_p.toString());
    }
    if (data.repetition_penalty !== undefined) {
      formData.append('repetition_penalty', data.repetition_penalty.toString());
    }
    if (data.seeds !== undefined) {
      formData.append('seeds', data.seeds.toString());
    }
    if (data.offloading) {
      formData.append('offloading', JSON.stringify(data.offloading));
    }

    const url = `${this.baseUrl}${this.transcriptionBase(projectId)}`;
    const locale = typeof window !== 'undefined'
      ? localStorage.getItem('vibevoice-locale') || 'en'
      : 'en';

    const response = await fetch(url, {
      method: 'POST',
      headers: {
        'X-Language': locale,
      },
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({
        error: 'Unknown error',
        message: response.statusText
      }));
      throw new Error(error.message || error.error || response.statusText);
    }

    return await response.json();
  }

  /**
   * Get a transcription with live progress
   */
  async getTranscription(projectId: string | null, requestId: string): Promise<Transcription> {
    return this.fetch(`${this.transcriptionBase(projectId)}/${encodeURIComponent(requestId)}`);
  }

  /**
   * Get the running transcription of this scope
   */
  async getCurrentTranscription(projectId: string | null): Promise<CurrentTranscriptionResponse> {
    return this.fetch(`${this.transcriptionBase(projectId)}/current`);
  }

  /**
   * List transcription history
   */
  async listTranscriptions(
    projectId: string | null,
    options?: { limit?: number; offset?: number }
  ): Promise<TranscriptionHistoryResponse> {
    const params = new URLSearchParams();
    if (options?.limit !== undefined) {
      params.append('limit', options.limit.toString());
    }
    if (options?.offset !== undefined) {
      params.append('offset', options.offset.toString());
    }

    const queryString = params.toString();
    const listPath = projectId ? this.transcriptionBase(projectId) : '/transcriptions/history';
    return this.fetch(`${listPath}${queryString ? '?' + queryString : ''}`);
  }

  /**
   * Delete a transcription
   */
  async deleteTranscription(
    projectId: string | null,
    requestId: string
  ): Promise<{ message: string; request_id: string }> {
    return this.fetch(`${this.transcriptionBase(projectId)}/${encodeURIComponent(requestId)}`, {
      method: 'DELETE',
    });
  }

  /**
   * Delete multiple transcriptions
   */
  async batchDeleteTranscriptions(
    projectId: string | null,
    requestIds: string[]
  ): Promise<BatchDeleteTranscriptionsResponse> {
    return this.fetch(`${this.transcriptionBase(projectId)}/batch-delete`, {
      method: 'POST',
      body: JSON.stringify({ request_ids: requestIds }),
    });
  }

  /**
   * Get URL of the source audio of a transcription
   */
  getTranscriptionAudioUrl(projectId: string | null, requestId: string): string {
    return `${this.baseUrl}${this.transcriptionBase(projectId)}/${encodeURIComponent(requestId)}/audio`;
  }

  /**
   * Get download URL for a transcript
   */
  getTranscriptionDownloadUrl(projectId: string | null, requestId: string, format: TranscriptFormat): string {
    return `${this.baseUrl}${this.transcriptionBase(projectId)}/${encodeURIComponent(requestId)}/download?format=${format}`;
  }
}

// Export singleton instance
export const api = new ApiClient();
