'use client';

import React, { useState, useMemo, useCallback } from 'react';
import { useProject } from '@/lib/ProjectContext';
import { useGeneration } from '@/lib/GenerationContext';
import { useLanguage } from '@/lib/i18n/LanguageContext';
import { api } from '@/lib/api';
import type { Generation } from '@/types/generation';
import { InferencePhase, getOffloadingMetrics } from '@/types/generation';
import toast from 'react-hot-toast';

function GenerationHistory() {
  const { currentProject } = useProject();
  const { generations, loading, deleteGeneration, batchDeleteGenerations } = useGeneration();
  const { t } = useLanguage();
  const [expandedId, setExpandedId] = useState<string | null>(null);

  // Pagination state
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(10);

  // Selection state
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());

  // Confirmation dialog state
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState<'single' | 'bulk'>('single');
  const [singleDeleteId, setSingleDeleteId] = useState<string | null>(null);

  const toggleDetails = useCallback((requestId: string) => {
    setExpandedId(prev => prev === requestId ? null : requestId);
  }, []);

  // Memoize pagination calculations
  const paginationData = useMemo(() => {
    const totalPages = Math.ceil(generations.length / itemsPerPage);
    const startIndex = (currentPage - 1) * itemsPerPage;
    const endIndex = startIndex + itemsPerPage;
    const paginatedGenerations = generations.slice(startIndex, endIndex);

    return {
      totalPages,
      startIndex,
      endIndex,
      paginatedGenerations
    };
  }, [generations, currentPage, itemsPerPage]);

  const { totalPages, startIndex, endIndex, paginatedGenerations } = paginationData;

  // Selection handlers
  const toggleSelection = useCallback((requestId: string) => {
    setSelectedIds(prev => {
      const newSelected = new Set(prev);
      if (newSelected.has(requestId)) {
        newSelected.delete(requestId);
      } else {
        newSelected.add(requestId);
      }
      return newSelected;
    });
  }, []);

  const toggleSelectAll = useCallback(() => {
    setSelectedIds(prev => {
      if (prev.size === paginatedGenerations.length) {
        return new Set();
      } else {
        return new Set(paginatedGenerations.map(g => g.request_id));
      }
    });
  }, [paginatedGenerations]);

  const isAllSelected = paginatedGenerations.length > 0 && selectedIds.size === paginatedGenerations.length;
  const isSomeSelected = selectedIds.size > 0 && selectedIds.size < paginatedGenerations.length;

  // Delete handlers
  const handleDeleteClick = useCallback((requestId: string) => {
    setSingleDeleteId(requestId);
    setDeleteTarget('single');
    setShowDeleteDialog(true);
  }, []);

  const handleBulkDeleteClick = useCallback(() => {
    setDeleteTarget('bulk');
    setShowDeleteDialog(true);
  }, []);

  const handleConfirmDelete = useCallback(async () => {
    try {
      if (deleteTarget === 'single' && singleDeleteId) {
        await deleteGeneration(singleDeleteId);
      } else if (deleteTarget === 'bulk') {
        await batchDeleteGenerations(Array.from(selectedIds));
        setSelectedIds(new Set());
      }
      setShowDeleteDialog(false);
      setSingleDeleteId(null);
    } catch (error) {
      console.error('Delete failed:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to delete generation(s)');
    }
  }, [deleteTarget, singleDeleteId, selectedIds, deleteGeneration, batchDeleteGenerations]);

  const handleCancelDelete = useCallback(() => {
    setShowDeleteDialog(false);
    setSingleDeleteId(null);
  }, []);

  // Pagination handlers
  const goToPage = useCallback((page: number) => {
    setCurrentPage(Math.max(1, Math.min(page, totalPages)));
  }, [totalPages]);

  const handleItemsPerPageChange = useCallback((value: number) => {
    setItemsPerPage(value);
    setCurrentPage(1); // Reset to first page when changing items per page
  }, []);

  const handleDownload = useCallback((generation: Generation) => {
    if (!currentProject || !generation.output_filename) return;

    // Add download=true parameter to force download
    const url = api.getGenerationDownloadUrl(currentProject.id, generation.request_id) + '?download=true';
    window.open(url, '_blank');
  }, [currentProject]);

  const getStatusColor = (status: InferencePhase): string => {
    switch (status) {
      case InferencePhase.COMPLETED:
        return 'bg-green-100 text-green-800';
      case InferencePhase.FAILED:
        return 'bg-red-100 text-red-800';
      case InferencePhase.PENDING:
        return 'bg-gray-100 text-gray-800';
      case InferencePhase.PREPROCESSING:
      case InferencePhase.INFERENCING:
      case InferencePhase.SAVING_AUDIO:
        return 'bg-blue-100 text-blue-800';
      default:
        return 'bg-gray-100 text-gray-800';
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

  const formatDate = (isoDate: string): string => {
    return new Date(isoDate).toLocaleString();
  };

  // Render generation details (similar to CurrentGeneration component)
  const renderGenerationDetails = (generation: Generation) => {
    const details = generation.details;
    const audioUrl = currentProject && generation.output_filename
      ? api.getGenerationDownloadUrl(currentProject.id, generation.request_id)
      : null;

    return (
      <div className="mt-4 pt-4 border-t border-gray-200 space-y-4">
        {/* Audio Player (for completed generations) */}
        {generation.status === InferencePhase.COMPLETED && audioUrl && (
          <div className="bg-gradient-to-r from-green-50 to-blue-50 rounded-lg p-4 border border-green-200">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center gap-2">
                <svg className="w-5 h-5 text-green-600" fill="currentColor" viewBox="0 0 20 20">
                  <path d="M18 3a1 1 0 00-1.196-.98l-10 2A1 1 0 006 5v9.114A4.369 4.369 0 005 14c-1.657 0-3 .895-3 2s1.343 2 3 2 3-.895 3-2V7.82l8-1.6v5.894A4.37 4.37 0 0015 12c-1.657 0-3 .895-3 2s1.343 2 3 2 3-.895 3-2V3z" />
                </svg>
                <label className="text-sm font-semibold text-gray-800">{t('generation.generatedAudio')}</label>
              </div>
              <a
                href={audioUrl + '?download=true'}
                download={generation.output_filename}
                className="px-3 py-1 text-xs bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors flex items-center gap-1"
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
            <div className="mt-2 flex items-center gap-4 text-xs text-gray-600">
              {details?.audio_duration_seconds && (
                <span>{t('generation.duration')}: {details.audio_duration_seconds.toFixed(2)}s</span>
              )}
              {generation.output_filename && (
                <span className="font-mono truncate">{generation.output_filename}</span>
              )}
            </div>
          </div>
        )}

        {/* Basic Information */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="text-xs font-medium text-gray-600">{t('generation.requestId')}</label>
            <p className="text-sm font-mono text-gray-900 break-all">{generation.request_id}</p>
          </div>
          <div>
            <label className="text-xs font-medium text-gray-600 block">{t('generation.sessionName')}</label>
            {(generation.session_name === t('session.deleted') || generation.session_name === t('session.unknown')) ? (
              <p className="text-sm text-gray-400 italic">{generation.session_name}</p>
            ) : (
              <a
                href={`/voice-editor?session=${generation.session_id}`}
                className="text-sm text-blue-600 hover:text-blue-800 hover:underline cursor-pointer block"
                title={t('generation.sessionId') + ': ' + generation.session_id}
              >
                {generation.session_name}
              </a>
            )}
          </div>
        </div>

        {/* Model Parameters */}
        <div>
          <label className="text-xs font-medium text-gray-600 block mb-2">{t('generation.modelParameters')}</label>
          <div className="grid grid-cols-2 gap-3 bg-gray-50 rounded p-3">
            <div>
              <p className="text-xs text-gray-600">{t('generation.cfgScale')}</p>
              <p className="text-sm font-semibold">{generation.cfg_scale ?? 'N/A'}</p>
            </div>
            <div>
              <p className="text-xs text-gray-600">{t('generation.seeds')}</p>
              <p className="text-sm font-semibold">{generation.seeds}</p>
            </div>
            <div>
              <p className="text-xs text-gray-600">{t('generation.attention')}</p>
              <p className="text-sm font-semibold">{generation.attn_implementation ?? 'N/A'}</p>
            </div>
            {generation.output_filename && (
              <div>
                <p className="text-xs text-gray-600">{t('generation.outputFile')}</p>
                <p className="text-sm font-semibold font-mono text-xs break-all">{generation.output_filename}</p>
              </div>
            )}
          </div>
        </div>

        {/* Phase-specific details */}
        {generation.status === InferencePhase.PREPROCESSING && details && (
          <div>
            <label className="text-xs font-medium text-gray-600 block mb-2">{t('generation.preprocessingInformation')}</label>
            {details.unique_speaker_names && (
              <div className="bg-blue-50 rounded p-3">
                <p className="text-xs font-medium mb-2">{t('generation.speakers')} ({details.unique_speaker_names.length})</p>
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
              <div className="bg-blue-50 rounded p-3 mt-2">
                <p className="text-xs">{t('generation.dialogLines')}: <span className="font-semibold">{details.scripts.length}</span></p>
              </div>
            )}
          </div>
        )}

        {(generation.status == InferencePhase.SAVING_AUDIO || generation.status == InferencePhase.COMPLETED) && details && (
          <div className="space-y-3">
            <label className="text-sm font-semibold text-gray-800 block border-b border-gray-300 pb-2">📊 {t('generation.generationStatistics')}</label>

            {/* Performance Summary Card */}
            {(details.generation_time !== undefined || details.audio_duration_seconds !== undefined || details.real_time_factor !== undefined) && (
              <div className="bg-gradient-to-br from-green-50 to-emerald-50 rounded-lg p-4 border border-green-200">
                <p className="text-xs font-semibold text-green-800 mb-3">⚡ {t('generation.performanceMetrics')}</p>
                <div className="grid grid-cols-3 gap-4">
                  {details.generation_time !== undefined && (
                    <div className="text-center">
                      <p className="text-2xl font-bold text-green-700">{details.generation_time.toFixed(2)}s</p>
                      <p className="text-xs text-gray-600 mt-1">{t('generation.generationTime')}</p>
                    </div>
                  )}
                  {details.audio_duration_seconds !== undefined && (
                    <div className="text-center">
                      <p className="text-2xl font-bold text-blue-700">{details.audio_duration_seconds.toFixed(2)}s</p>
                      <p className="text-xs text-gray-600 mt-1">{t('generation.audioDuration')}</p>
                    </div>
                  )}
                  {details.real_time_factor !== undefined && (
                    <div className="text-center">
                      <p className="text-2xl font-bold text-purple-700">{details.real_time_factor.toFixed(2)}×</p>
                      <p className="text-xs text-gray-600 mt-1">{t('generation.realTimeFactor')}</p>
                    </div>
                  )}
                </div>
                {details.number_of_segments !== undefined && (
                  <div className="mt-3 pt-3 border-t border-green-200 text-center">
                    <p className="text-xs text-gray-600">
                      {t('generation.generatedSegments').replace('{count}', details.number_of_segments.toString())}
                    </p>
                  </div>
                )}
              </div>
            )}

            {/* Token Statistics */}
            {(details.prefilling_tokens !== undefined || details.generated_tokens !== undefined || details.total_tokens !== undefined) && (
              <div className="bg-blue-50 rounded-lg p-4 border border-blue-200">
                <p className="text-xs font-semibold text-blue-800 mb-3">🔢 {t('generation.tokenStatistics')}</p>
                <div className="grid grid-cols-3 gap-4">
                  {details.prefilling_tokens !== undefined && (
                    <div className="text-center">
                      <p className="text-xl font-bold text-blue-700">{details.prefilling_tokens.toLocaleString()}</p>
                      <p className="text-xs text-gray-600 mt-1">{t('generation.prefilling')}</p>
                    </div>
                  )}
                  {details.generated_tokens !== undefined && (
                    <div className="text-center">
                      <p className="text-xl font-bold text-blue-700">{details.generated_tokens.toLocaleString()}</p>
                      <p className="text-xs text-gray-600 mt-1">{t('generation.generated')}</p>
                    </div>
                  )}
                  {details.total_tokens !== undefined && (
                    <div className="text-center">
                      <p className="text-xl font-bold text-blue-700">{details.total_tokens.toLocaleString()}</p>
                      <p className="text-xs text-gray-600 mt-1">{t('generation.totalTokens')}</p>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Offloading Metrics */}
            {(() => {
              const metrics = getOffloadingMetrics(generation);
              if (metrics) {
                return (
                  <div className="bg-gradient-to-br from-orange-50 to-amber-50 rounded-lg p-4 border border-orange-200">
                    <div className="flex items-center gap-2 mb-3">
                      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20" fill="currentColor" className="w-5 h-5 text-orange-700">
                        <path fillRule="evenodd" d="M7.84 1.804A1 1 0 018.82 1h2.36a1 1 0 01.98.804l.331 1.652a6.993 6.993 0 011.929 1.115l1.598-.54a1 1 0 011.186.447l1.18 2.044a1 1 0 01-.205 1.251l-1.267 1.113a7.047 7.047 0 010 2.228l1.267 1.113a1 1 0 01.206 1.25l-1.18 2.045a1 1 0 01-1.187.447l-1.598-.54a6.993 6.993 0 01-1.929 1.115l-.33 1.652a1 1 0 01-.98.804H8.82a1 1 0 01-.98-.804l-.331-1.652a6.993 6.993 0 01-1.929-1.115l-1.598.54a1 1 0 01-1.186-.447l-1.18-2.044a1 1 0 01.205-1.251l1.267-1.114a7.05 7.05 0 010-2.227L1.821 7.773a1 1 0 01-.206-1.25l1.18-2.045a1 1 0 011.187-.447l1.598.54A6.993 6.993 0 017.51 3.456l.33-1.652zM10 13a3 3 0 100-6 3 3 0 000 6z" clipRule="evenodd" />
                      </svg>
                      <p className="text-xs font-semibold text-orange-800">⚙️ {t('generation.offloadingMetrics')}</p>
                    </div>

                    <div className="space-y-3">
                      {/* Configuration */}
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-700">{t('generation.configuration')}:</span>
                        <span className="font-semibold text-orange-900">{metrics.gpu_layers} GPU / {metrics.cpu_layers} CPU layers</span>
                      </div>
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-700">{t('generation.vramSaved')}:</span>
                        <span className="font-semibold text-orange-900">~{metrics.vram_saved_gb} GB</span>
                      </div>

                      {/* Performance Breakdown */}
                      <div className="border-t border-orange-200 pt-3">
                        <div className="font-semibold text-xs text-orange-800 mb-2">{t('generation.performanceBreakdown')}:</div>
                        <div className="space-y-2">
                          <div className="flex justify-between text-xs">
                            <span className="text-gray-700">{t('generation.pureComputation')}:</span>
                            <span className="font-mono text-gray-900">
                              {(metrics.time_breakdown.pure_computation_ms / 1000).toFixed(2)}s
                              <span className="text-gray-500 ml-1">
                                ({((metrics.time_breakdown.pure_computation_ms / metrics.transfer_overhead_ms) * 100).toFixed(1)}%)
                              </span>
                            </span>
                          </div>
                          <div className="flex justify-between text-xs">
                            <span className="text-gray-700">{t('generation.cpuToGpuTransfers')}:</span>
                            <span className="font-mono text-gray-900">
                              {(metrics.time_breakdown.cpu_to_gpu_transfer_ms / 1000).toFixed(2)}s
                              <span className="text-gray-500 ml-1">
                                ({((metrics.time_breakdown.cpu_to_gpu_transfer_ms / metrics.transfer_overhead_ms) * 100).toFixed(1)}%)
                              </span>
                            </span>
                          </div>
                          <div className="flex justify-between text-xs">
                            <span className="text-gray-700">{t('generation.gpuToCpuReleases')}:</span>
                            <span className="font-mono text-gray-900">
                              {(metrics.time_breakdown.gpu_to_cpu_release_ms / 1000).toFixed(2)}s
                              <span className="text-gray-500 ml-1">
                                ({((metrics.time_breakdown.gpu_to_cpu_release_ms / metrics.transfer_overhead_ms) * 100).toFixed(1)}%)
                              </span>
                            </span>
                          </div>
                          <div className="flex justify-between text-xs font-semibold border-t border-orange-200 pt-2">
                            <span className="text-gray-700">{t('generation.transferOverhead')}:</span>
                            <span className="text-orange-900">{metrics.overhead_percentage.toFixed(1)}%</span>
                          </div>
                        </div>
                      </div>

                      {/* Theoretical Async Savings */}
                      {metrics.theoretical_async_savings_ms > 0 && (
                        <div className="border-t border-orange-200 pt-3">
                          <div className="text-xs text-gray-600 bg-white bg-opacity-50 rounded p-2">
                            💡 Transfer time is PCIe bandwidth limited. Future async prefetching could save
                            <span className="font-semibold text-orange-900 ml-1">
                              ~{(metrics.theoretical_async_savings_ms / 1000).toFixed(1)}s
                            </span>
                            <span className="text-gray-500 ml-1">
                              ({((metrics.theoretical_async_savings_ms / metrics.transfer_overhead_ms) * 100).toFixed(1)}% faster)
                            </span>
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                );
              }
              return null;
            })()}

            {/* Speakers */}
            {details.unique_speaker_names && (
              <div className="bg-purple-50 rounded-lg p-4 border border-purple-200">
                <p className="text-xs font-semibold text-purple-800 mb-2">🎤 {t('generation.speakersUsed').replace('{count}', details.unique_speaker_names.length.toString())}</p>
                <div className="flex flex-wrap gap-2">
                  {details.unique_speaker_names.map((speaker: string, idx: number) => (
                    <span key={idx} className="px-3 py-1.5 bg-purple-100 text-purple-900 rounded-full text-xs font-medium">
                      {speaker}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Output Path */}
            {details.output_audio_path && (
              <div className="bg-gray-100 rounded-lg p-4 border border-gray-300">
                <p className="text-xs font-semibold text-gray-700 mb-2">📁 {t('generation.outputPath')}</p>
                <p className="text-xs font-mono break-all text-gray-600 bg-white p-2 rounded border border-gray-200">{details.output_audio_path}</p>
              </div>
            )}
          </div>
        )}

        {generation.status === InferencePhase.FAILED && details && (
          <div>
            <label className="text-xs font-medium text-gray-600 block mb-2">{t('generation.errorInformation')}</label>
            <div className="bg-red-50 rounded p-3">
              <pre className="text-xs whitespace-pre-wrap text-red-900">
                {details.error || JSON.stringify(details, null, 2)}
              </pre>
            </div>
          </div>
        )}

        {/* Timestamps */}
        <div className="bg-gray-50 rounded p-3">
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div>
              <p className="text-gray-600">{t('generation.created')}</p>
              <p className="font-medium">{formatDate(generation.created_at)}</p>
            </div>
            <div>
              <p className="text-gray-600">{t('generation.updated')}</p>
              <p className="font-medium">{formatDate(generation.updated_at)}</p>
            </div>
          </div>
        </div>
      </div>
    );
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-gray-500">{t('generation.loadingHistory')}</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full">
      {/* Header with selection controls */}
      <div className="mb-4">
        <div className="flex items-center justify-between mb-2">
          <h2 className="text-xl font-semibold">{t('generation.generationHistory')}</h2>
          {selectedIds.size > 0 && (
            <button
              onClick={handleBulkDeleteClick}
              className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600 flex items-center gap-2"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
              </svg>
              {t('generation.deleteSelected').replace('{count}', selectedIds.size.toString())}
            </button>
          )}
        </div>
        <p className="text-sm text-gray-600">
          {t('generation.totalGenerations')
            .replace('{count}', generations.length.toString())
            .replace('{plural}', generations.length !== 1 ? 's' : '')}
          {selectedIds.size > 0 && ` ${t('generation.itemsSelected').replace('{count}', selectedIds.size.toString())}`}
        </p>
      </div>

      {generations.length === 0 ? (
        <div className="flex items-center justify-center flex-1 text-gray-500">
          <p>{t('generation.noHistory')}</p>
        </div>
      ) : (
        <>
          {/* Select all checkbox */}
          {paginatedGenerations.length > 0 && (
            <div className="mb-2 flex items-center gap-2 px-2">
              <input
                type="checkbox"
                checked={isAllSelected}
                ref={(el) => {
                  if (el) el.indeterminate = isSomeSelected;
                }}
                onChange={toggleSelectAll}
                className="w-4 h-4 text-blue-600 rounded border-gray-300 focus:ring-blue-500"
              />
              <label className="text-sm text-gray-700 cursor-pointer" onClick={toggleSelectAll}>
                {isAllSelected ? t('generation.deselectAll') : t('generation.selectAllOnPage')}
              </label>
            </div>
          )}

          {/* Generation list */}
          <div className="space-y-2 overflow-y-auto flex-1">
            {paginatedGenerations.map((generation) => {
              const isExpanded = expandedId === generation.request_id;
              const isSelected = selectedIds.has(generation.request_id);

              return (
                <div
                  key={generation.request_id}
                  className={`border rounded-lg bg-white transition-all ${
                    isSelected ? 'border-blue-500 ring-2 ring-blue-200' : 'border-gray-300'
                  }`}
                >
                  <div className="p-4">
                    <div className="flex items-start gap-3">
                      {/* Selection checkbox */}
                      <div className="pt-1">
                        <input
                          type="checkbox"
                          checked={isSelected}
                          onChange={() => toggleSelection(generation.request_id)}
                          className="w-4 h-4 text-blue-600 rounded border-gray-300 focus:ring-blue-500"
                          onClick={(e) => e.stopPropagation()}
                        />
                      </div>

                      {/* Main content */}
                      <div className="flex-1">
                        <div className="flex items-center gap-2 mb-2">
                          <span
                            className={`px-2 py-1 rounded text-xs font-medium ${getStatusColor(
                              generation.status
                            )}`}
                          >
                            {getStatusLabel(generation.status)}
                          </span>
                          <span className="text-xs text-gray-500">
                            {generation.model_dtype}
                          </span>
                        </div>

                        <p className="text-sm text-gray-600 mb-1">
                          {t('generation.session')}: <span className="font-medium">{generation.session_id}</span>
                        </p>

                        <p className="text-xs text-gray-500">
                          {t('generation.created')}: {formatDate(generation.created_at)}
                        </p>

                        {generation.percentage !== null && (
                          <div className="mt-2">
                            <div className="w-full bg-gray-200 rounded-full h-2">
                              <div
                                className="bg-blue-500 h-2 rounded-full transition-all"
                                style={{ width: `${generation.percentage}%` }}
                              />
                            </div>
                            <p className="text-xs text-gray-500 mt-1">
                              {generation.percentage.toFixed(1)}%
                            </p>
                          </div>
                        )}
                      </div>

                      {/* Action buttons */}
                      <div className="flex flex-col gap-2">
                        <button
                          onClick={() => toggleDetails(generation.request_id)}
                          className="w-32 px-3 py-1 text-sm bg-gray-200 text-gray-700 rounded hover:bg-gray-300 flex items-center justify-center gap-1"
                        >
                          {isExpanded ? (
                            <>
                              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" />
                              </svg>
                              <span>{t('generation.hide')}</span>
                            </>
                          ) : (
                            <>
                              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                              </svg>
                              <span>{t('generation.viewDetails')}</span>
                            </>
                          )}
                        </button>

                        {generation.status === InferencePhase.COMPLETED && generation.output_filename && (
                          <button
                            onClick={() => handleDownload(generation)}
                            className="w-32 px-3 py-1 text-sm bg-blue-500 text-white rounded hover:bg-blue-600 flex items-center justify-center gap-1"
                          >
                            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                            </svg>
                            <span>{t('generation.download')}</span>
                          </button>
                        )}

                        <button
                          onClick={() => handleDeleteClick(generation.request_id)}
                          className="w-32 px-3 py-1 text-sm bg-red-500 text-white rounded hover:bg-red-600 flex items-center justify-center gap-1"
                        >
                          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                          </svg>
                          <span>{t('common.delete')}</span>
                        </button>
                      </div>
                    </div>

                    {/* Expandable Details Section */}
                    {isExpanded && renderGenerationDetails(generation)}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Pagination controls */}
          <div className="mt-4 flex items-center justify-between border-t border-gray-300 pt-4">
            <div className="flex items-center gap-2">
              <label className="text-sm text-gray-700">{t('generation.itemsPerPage')}:</label>
              <select
                value={itemsPerPage}
                onChange={(e) => handleItemsPerPageChange(Number(e.target.value))}
                className="border border-gray-300 rounded px-2 py-1 text-sm"
              >
                <option value={5}>5</option>
                <option value={10}>10</option>
                <option value={20}>20</option>
                <option value={50}>50</option>
              </select>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-700">
                {t('generation.page')} {currentPage} {t('generation.of')} {totalPages} ({t('generation.showingItems')
                  .replace('{start}', (startIndex + 1).toString())
                  .replace('{end}', Math.min(endIndex, generations.length).toString())
                  .replace('{total}', generations.length.toString())})
              </span>
              <div className="flex gap-1">
                <button
                  onClick={() => goToPage(1)}
                  disabled={currentPage === 1}
                  className="px-2 py-1 border border-gray-300 rounded text-sm disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-100"
                >
                  ««
                </button>
                <button
                  onClick={() => goToPage(currentPage - 1)}
                  disabled={currentPage === 1}
                  className="px-2 py-1 border border-gray-300 rounded text-sm disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-100"
                >
                  ‹
                </button>
                <button
                  onClick={() => goToPage(currentPage + 1)}
                  disabled={currentPage === totalPages}
                  className="px-2 py-1 border border-gray-300 rounded text-sm disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-100"
                >
                  ›
                </button>
                <button
                  onClick={() => goToPage(totalPages)}
                  disabled={currentPage === totalPages}
                  className="px-2 py-1 border border-gray-300 rounded text-sm disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-100"
                >
                  »»
                </button>
              </div>
            </div>
          </div>
        </>
      )}

      {/* Confirmation Dialog */}
      {showDeleteDialog && (
        <div className="fixed inset-0 bg-gray-900/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4 shadow-xl">
            <h3 className="text-lg font-semibold mb-2">{t('generation.confirmDeletionTitle')}</h3>
            <p className="text-gray-600 mb-4">
              {deleteTarget === 'single'
                ? t('generation.confirmSingleDelete')
                : t('generation.confirmBulkDelete')
                    .replace('{count}', selectedIds.size.toString())
                    .replace('{plural}', selectedIds.size > 1 ? 's' : '')}
            </p>
            <div className="flex gap-3 justify-end">
              <button
                onClick={handleCancelDelete}
                className="px-4 py-2 border border-gray-300 rounded-lg hover:bg-gray-100"
              >
                {t('common.cancel')}
              </button>
              <button
                onClick={handleConfirmDelete}
                className="px-4 py-2 bg-red-500 text-white rounded-lg hover:bg-red-600"
              >
                {t('common.delete')}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// Export with React.memo to prevent unnecessary re-renders
export default React.memo(GenerationHistory);
