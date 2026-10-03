'use client';

import React, { useState, useCallback, useEffect } from 'react';
import { useLanguage } from '@/lib/i18n/LanguageContext';
import { api } from '@/lib/api';
import type { TranscriptionSummary } from '@/types/transcription';
import { isTranscriptionActive } from '@/types/transcription';
import toast from 'react-hot-toast';

interface TranscriptionHistoryProps {
  projectId: string | null;
  onSelect: (requestId: string) => void;
  currentId?: string;
  currentStatus?: string;
}

export default function TranscriptionHistory({ projectId, onSelect, currentId, currentStatus }: TranscriptionHistoryProps) {
  const { t } = useLanguage();
  const [items, setItems] = useState<TranscriptionSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage] = useState(10);
  const [total, setTotal] = useState(0);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [deleteIds, setDeleteIds] = useState<string[] | null>(null);

  const loadHistory = useCallback(async () => {
    try {
      setLoading(true);
      const offset = (currentPage - 1) * itemsPerPage;
      const response = await api.listTranscriptions(projectId, { offset, limit: itemsPerPage });
      setItems(response.transcriptions);
      setTotal(response.total);
    } catch (err) {
      console.error('Failed to load transcription history:', err);
      toast.error(t('transcription.errorLoadHistory'));
    } finally {
      setLoading(false);
    }
  }, [projectId, currentPage, itemsPerPage, t]);

  useEffect(() => {
    loadHistory();
  }, [loadHistory]);

  useEffect(() => {
    if (currentId) {
      loadHistory();
    }
  }, [currentId, loadHistory]);

  useEffect(() => {
    if (currentStatus === 'completed' || currentStatus === 'failed') {
      loadHistory();
    }
  }, [currentStatus, loadHistory]);

  const totalPages = Math.ceil(total / itemsPerPage);
  const isAllSelected = items.length > 0 && selectedIds.size === items.length;

  const toggleSelection = useCallback((requestId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setSelectedIds(prev => {
      const next = new Set(prev);
      if (next.has(requestId)) {
        next.delete(requestId);
      } else {
        next.add(requestId);
      }
      return next;
    });
  }, []);

  const toggleSelectAll = useCallback(() => {
    setSelectedIds(prev => (prev.size === items.length ? new Set() : new Set(items.map(i => i.request_id))));
  }, [items]);

  const handleConfirmDelete = useCallback(async () => {
    if (!deleteIds) return;
    try {
      if (deleteIds.length === 1) {
        await api.deleteTranscription(projectId, deleteIds[0]);
        toast.success(t('transcription.deleted'));
      } else {
        const result = await api.batchDeleteTranscriptions(projectId, deleteIds);
        if (result.failed_count > 0) {
          toast.error(t('transcription.bulkDeletePartial', {
            deleted: result.deleted_count,
            failed: result.failed_count,
          }));
        } else {
          toast.success(t('transcription.bulkDeleted', { count: result.deleted_count }));
        }
      }
      setSelectedIds(new Set());
      await loadHistory();
    } catch (err) {
      toast.error(err instanceof Error ? err.message : t('transcription.errorDelete'));
    } finally {
      setDeleteIds(null);
    }
  }, [deleteIds, projectId, loadHistory, t]);

  const formatDuration = (seconds: number | null): string => {
    if (seconds === null || seconds === undefined) return '-';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
  };

  const getStatusBadge = (status: string): string => {
    switch (status) {
      case 'completed':
        return 'bg-green-100 text-green-800';
      case 'failed':
        return 'bg-red-100 text-red-800';
      case 'preprocessing':
      case 'inferencing':
        return 'bg-blue-100 text-blue-800';
      default:
        return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold text-gray-900">
          {t('transcription.history')}
          <span className="ml-2 text-sm font-normal text-gray-500">({total})</span>
        </h2>
        <div className="flex items-center gap-2">
          {selectedIds.size > 0 && (
            <button
              onClick={() => setDeleteIds(Array.from(selectedIds))}
              className="px-3 py-1.5 text-sm bg-red-500 text-white rounded-lg hover:bg-red-600"
            >
              {t('transcription.deleteSelected', { count: selectedIds.size })}
            </button>
          )}
          <button
            onClick={loadHistory}
            className="px-3 py-1.5 text-sm border border-gray-300 rounded-lg hover:bg-gray-100"
          >
            {t('common.refresh')}
          </button>
        </div>
      </div>

      {items.length > 0 && (
        <label className="flex items-center gap-2 mb-2 text-sm text-gray-600 cursor-pointer">
          <input
            type="checkbox"
            checked={isAllSelected}
            onChange={toggleSelectAll}
            className="w-4 h-4 rounded"
          />
          {t('common.selectAll')}
        </label>
      )}

      <div className="flex-1 overflow-y-auto space-y-2">
        {loading && items.length === 0 ? (
          <div className="flex justify-center py-8">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600" />
          </div>
        ) : items.length === 0 ? (
          <div className="text-center py-12 text-gray-500">
            <svg className="w-12 h-12 mx-auto mb-3 text-gray-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            <p className="font-medium">{t('transcription.noHistory')}</p>
            <p className="text-sm mt-1">{t('transcription.noHistoryHint')}</p>
          </div>
        ) : (
          items.map(item => {
            const isCurrent = item.request_id === currentId;
            const running = isTranscriptionActive(item.status);
            return (
              <div
                key={item.request_id}
                onClick={() => onSelect(item.request_id)}
                className={`p-3 border rounded-lg cursor-pointer transition-colors ${
                  isCurrent ? 'border-blue-400 bg-blue-50' : 'border-gray-200 hover:bg-gray-50'
                }`}
              >
                <div className="flex items-start gap-3">
                  <input
                    type="checkbox"
                    checked={selectedIds.has(item.request_id)}
                    onClick={(e) => toggleSelection(item.request_id, e)}
                    onChange={() => {}}
                    disabled={running}
                    className="mt-1 w-4 h-4 rounded"
                  />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between gap-2">
                      <p className="text-sm font-medium text-gray-900 truncate">{item.original_filename}</p>
                      <span className={`px-2 py-0.5 text-xs font-medium rounded-full flex-shrink-0 ${getStatusBadge(item.status)}`}>
                        {t(`transcription.status.${item.status}`)}
                      </span>
                    </div>
                    {item.text_preview && (
                      <p className="text-xs text-gray-600 mt-1 line-clamp-2">{item.text_preview}</p>
                    )}
                    <div className="flex items-center gap-3 mt-1 text-xs text-gray-500">
                      <span>{new Date(item.created_at).toLocaleString()}</span>
                      <span>{formatDuration(item.audio_duration)}</span>
                      {item.segment_count > 0 && (
                        <span>{t('transcription.segmentCount', { count: item.segment_count })}</span>
                      )}
                      <span>{item.model_dtype}</span>
                    </div>
                  </div>
                  {!running && (
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        setDeleteIds([item.request_id]);
                      }}
                      className="p-1 text-gray-400 hover:text-red-500"
                      title={t('common.delete')}
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                    </button>
                  )}
                </div>
              </div>
            );
          })
        )}
      </div>

      {totalPages > 1 && (
        <div className="flex items-center justify-between pt-3 mt-3 border-t border-gray-200 text-sm">
          <button
            onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
            disabled={currentPage === 1}
            className="px-3 py-1 border border-gray-300 rounded-lg disabled:opacity-50"
          >
            {t('common.previous')}
          </button>
          <span className="text-gray-600">{currentPage} / {totalPages}</span>
          <button
            onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))}
            disabled={currentPage === totalPages}
            className="px-3 py-1 border border-gray-300 rounded-lg disabled:opacity-50"
          >
            {t('common.next')}
          </button>
        </div>
      )}

      {deleteIds && (
        <div className="fixed inset-0 bg-gray-900/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4 shadow-xl">
            <h3 className="text-lg font-semibold mb-2">{t('transcription.confirmDeletionTitle')}</h3>
            <p className="text-gray-600 mb-4">
              {deleteIds.length === 1
                ? t('transcription.confirmSingleDelete')
                : t('transcription.confirmBulkDelete', { count: deleteIds.length })}
            </p>
            <div className="flex gap-3 justify-end">
              <button
                onClick={() => setDeleteIds(null)}
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
