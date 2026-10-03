/**
 * Unified Task types for the Task API
 */

import type { Generation } from './generation';
import type { TrainingState } from './training';
import type { QuickGenerate } from './quickGenerate';
import type { Transcription } from './transcription';

/**
 * Task type enum
 */
export type TaskType = 'inference' | 'training' | 'quick_generation' | 'transcription' | null;

/**
 * Unified task data - can be inference (Generation), training (TrainingState), quick generation (QuickGenerate) or transcription (Transcription)
 */
export type TaskData = Generation | TrainingState | QuickGenerate | Transcription | null;

/**
 * Unified task response from GET /api/v1/tasks/current
 */
export interface CurrentTask {
  type: TaskType;
  project_id: string | null;
  data: TaskData;
}

/**
 * Response from GET /api/v1/tasks/current
 */
export interface CurrentTaskResponse {
  message: string;
  task: CurrentTask;
}

/**
 * Type guard to check if task data is a Generation (inference task)
 */
export function isInferenceTask(task: CurrentTask): task is CurrentTask & { type: 'inference'; data: Generation } {
  return task.type === 'inference' && task.data !== null;
}

/**
 * Type guard to check if task data is a TrainingState (training task)
 */
export function isTrainingTask(task: CurrentTask): task is CurrentTask & { type: 'training'; data: TrainingState } {
  return task.type === 'training' && task.data !== null;
}

/**
 * Type guard to check if task data is a QuickGenerate (quick generation task)
 */
export function isQuickGenerationTask(task: CurrentTask): task is CurrentTask & { type: 'quick_generation'; data: QuickGenerate } {
  return task.type === 'quick_generation' && task.data !== null;
}

/**
 * Type guard to check if task data is a Transcription (ASR task)
 */
export function isTranscriptionTask(task: CurrentTask): task is CurrentTask & { type: 'transcription'; data: Transcription } {
  return task.type === 'transcription' && task.data !== null;
}

/**
 * Check if there's any active task
 */
export function hasActiveTask(task: CurrentTask | null): boolean {
  return task !== null && task.type !== null && task.data !== null;
}
