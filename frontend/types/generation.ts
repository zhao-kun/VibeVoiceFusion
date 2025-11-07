/**
 * Generation types for Voice Generation API
 */

/**
 * Inference phase enum matching backend InferencePhase
 * Note: Backend currently returns lowercase values
 */
export enum InferencePhase {
  PENDING = 'pending',
  PREPROCESSING = 'preprocessing',
  INFERENCING = 'inferencing',
  SAVING_AUDIO = 'saving_audio',
  COMPLETED = 'completed',
  FAILED = 'failed'
}

/**
 * Model dtype options
 */
export type ModelDtype = 'bf16' | 'float8_e4m3fn';

/**
 * Generation metadata from backend
 */
export interface Generation {
  request_id: string;
  session_id: string;
  session_name: string;
  status: InferencePhase;
  output_filename: string | null;
  percentage: number | null;
  model_dtype: ModelDtype;
  cfg_scale: number | null;
  attn_implementation: string | null;
  seeds: number;
  details: Record<string, any>;
  created_at: string;
  updated_at: string;
  project_id?: string | null;
  project_dir?: string | null;
}

/**
 * Offloading configuration mode
 */
export type OffloadingMode = 'preset' | 'manual';

/**
 * Offloading preset options
 */
export type OffloadingPreset = 'balanced' | 'aggressive' | 'extreme';

/**
 * Offloading configuration for generation request
 */
export interface OffloadingConfig {
  enabled: boolean;
  mode: OffloadingMode;
  preset?: OffloadingPreset;      // Required for preset mode
  num_gpu_layers?: number;        // Required for manual mode (1-28)
}

/**
 * Time breakdown for offloading metrics
 */
export interface OffloadingTimeBreakdown {
  pure_computation_ms: number;
  cpu_to_gpu_transfer_ms: number;
  gpu_to_cpu_release_ms: number;
}

/**
 * Offloading metrics returned after generation completes
 */
export interface OffloadingMetrics {
  enabled: boolean;
  gpu_layers: number;
  cpu_layers: number;
  transfer_overhead_ms: number;
  avg_layer_transfer_ms: number;
  overhead_percentage: number;
  time_breakdown: OffloadingTimeBreakdown;
  theoretical_async_savings_ms: number;
  vram_saved_gb: number;
}

/**
 * Request body for creating generation
 */
export interface CreateGenerationRequest {
  dialog_session_id: string;
  seeds?: number;
  cfg_scale?: number;
  model_dtype?: ModelDtype;
  attn_implementation?: string;
  offloading?: OffloadingConfig;  // NEW: Optional offloading configuration
}

/**
 * Response from POST /generations
 */
export interface CreateGenerationResponse {
  message: string;
  request_id: string;
  generation: Generation;
}

/**
 * Response from GET /generations/current
 * Note: generation is null when no active generation (200 response, not an error)
 */
export interface CurrentGenerationResponse {
  message: string;
  generation: Generation | null;
}

/**
 * Response from GET /generations
 */
export interface ListGenerationsResponse {
  generations: Generation[];
  count: number;
}

/**
 * Response from GET /generations/:request_id
 */
export interface GetGenerationResponse {
  generation: Generation;
}

/**
 * Helper function to extract offloading configuration from generation.details
 */
export function getOffloadingConfig(generation: Generation): OffloadingConfig | null {
  return generation.details?.offloading_config || null;
}

/**
 * Helper function to extract offloading metrics from generation.details
 */
export function getOffloadingMetrics(generation: Generation): OffloadingMetrics | null {
  return generation.details?.offloading_metrics || null;
}
