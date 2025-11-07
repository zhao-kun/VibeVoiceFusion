"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useSession } from "@/lib/SessionContext";
import { useLanguage } from "@/lib/i18n/LanguageContext";
import toast from "react-hot-toast";

export default function SessionManager() {
  const router = useRouter();
  const { sessions, currentSession, selectSession, createSession, deleteSession, updateSession, loading, error } = useSession();
  const { t } = useLanguage();
  const [showCreateModal, setShowCreateModal] = useState(false);
  const [showEditModal, setShowEditModal] = useState(false);
  const [showDeleteDialog, setShowDeleteDialog] = useState(false);
  const [editingSession, setEditingSession] = useState<string | null>(null);
  const [deleteTargetId, setDeleteTargetId] = useState<string | null>(null);
  const [newSessionName, setNewSessionName] = useState("");
  const [newSessionDescription, setNewSessionDescription] = useState("");
  const [localError, setLocalError] = useState<string | null>(null);

  const handleCreateSession = async () => {
    if (newSessionName.trim()) {
      setLocalError(null);
      try {
        await createSession(newSessionName.trim(), newSessionDescription.trim());
        setNewSessionName("");
        setNewSessionDescription("");
        setShowCreateModal(false);
      } catch (err) {
        setLocalError(err instanceof Error ? err.message : t('session.createError'));
      }
    }
  };

  const handleEditSession = (sessionId: string) => {
    const session = sessions.find((s) => s.id === sessionId);
    if (session) {
      setEditingSession(sessionId);
      setNewSessionName(session.name);
      setNewSessionDescription(session.description);
      setShowEditModal(true);
    }
  };

  const handleUpdateSession = async () => {
    if (editingSession && newSessionName.trim()) {
      setLocalError(null);
      try {
        await updateSession(editingSession, {
          name: newSessionName.trim(),
          description: newSessionDescription.trim(),
        });
        setEditingSession(null);
        setNewSessionName("");
        setNewSessionDescription("");
        setShowEditModal(false);
      } catch (err) {
        setLocalError(err instanceof Error ? err.message : t('session.updateError'));
      }
    }
  };

  const handleDeleteClick = (sessionId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setDeleteTargetId(sessionId);
    setShowDeleteDialog(true);
  };

  const handleConfirmDelete = async () => {
    if (!deleteTargetId) return;

    setLocalError(null);
    try {
      await deleteSession(deleteTargetId);
      toast.success(t('session.deleteSuccess'));
    } catch (err) {
      setLocalError(err instanceof Error ? err.message : t('session.deleteError'));
      toast.error(err instanceof Error ? err.message : t('session.deleteError'));
    } finally {
      setShowDeleteDialog(false);
      setDeleteTargetId(null);
    }
  };

  const handleCancelDelete = () => {
    setShowDeleteDialog(false);
    setDeleteTargetId(null);
  };

  return (
    <div className="h-full flex flex-col bg-gray-50 border-r border-gray-200">
      {/* Error Display */}
      {(error || localError) && (
        <div className="bg-red-50 border-b border-red-200 px-4 py-2">
          <p className="text-xs text-red-800">
            {error || localError}
          </p>
        </div>
      )}

      <div className="p-4 border-b border-gray-200">
        <h2 className="text-lg font-semibold text-gray-800">{t('session.title')}</h2>
        <p className="text-xs text-gray-500 mt-1">{t('session.manageMultipleSessions')}</p>
      </div>

      <div className="flex-1 overflow-y-auto p-2">
        {sessions.map((session) => {
          const isActive = currentSession?.id === session.id;

          return (
            <div
              key={session.id}
              onClick={() => router.push(`/voice-editor?session=${session.id}`)}
              className={`
                p-3 mb-2 rounded-lg cursor-pointer transition-all duration-150
                ${
                  isActive
                    ? "bg-blue-500 text-white shadow-md"
                    : "bg-white text-gray-700 hover:bg-gray-100 hover:shadow-sm"
                }
              `}
            >
              <div className="flex items-start justify-between mb-2">
                <div className="flex-1 min-w-0">
                  <div className="font-medium truncate">{session.name}</div>
                  {session.description && (
                    <p className={`text-xs mt-1 line-clamp-2 ${isActive ? "text-blue-100" : "text-gray-500"}`}>
                      {session.description}
                    </p>
                  )}
                </div>
                <div className="flex items-center space-x-1 ml-2">
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      handleEditSession(session.id);
                    }}
                    className={`p-1 rounded hover:bg-white/20 ${isActive ? "text-white" : "text-gray-400 hover:text-blue-600"}`}
                    title={t('session.editSessionTooltip')}
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                    </svg>
                  </button>
                  {sessions.length > 1 && (
                    <button
                      onClick={(e) => handleDeleteClick(session.id, e)}
                      className={`p-1 rounded hover:bg-red-100 ${isActive ? "text-white hover:text-red-600" : "text-gray-400 hover:text-red-600"}`}
                      title={t('session.deleteSessionTooltip')}
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                      </svg>
                    </button>
                  )}
                </div>
              </div>

              <div className={`flex items-center text-xs ${isActive ? "text-blue-100" : "text-gray-500"}`}>
                <svg className="w-3 h-3 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
                <span>{session.dialogLines.length} {t('session.lines')}</span>
              </div>
            </div>
          );
        })}
      </div>

      <div className="p-4 border-t border-gray-200">
        <button
          onClick={() => setShowCreateModal(true)}
          disabled={loading}
          className="w-full px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors flex items-center justify-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
          </svg>
          <span>{loading ? t('common.loading') : t('session.newSession')}</span>
        </button>
      </div>

      {/* Create Session Modal */}
      {showCreateModal && (
        <div className="fixed inset-0 bg-gray-900/50 flex items-center justify-center z-[200] p-4">
          <div className="bg-white rounded-xl shadow-xl max-w-md w-full p-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">{t('session.createNewSession')}</h2>

            <div className="space-y-4 mb-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  {t('session.sessionName')} *
                </label>
                <input
                  type="text"
                  value={newSessionName}
                  onChange={(e) => setNewSessionName(e.target.value)}
                  placeholder={t('session.enterSessionName')}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  autoFocus
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  {t('project.projectDescription')}
                </label>
                <textarea
                  value={newSessionDescription}
                  onChange={(e) => setNewSessionDescription(e.target.value)}
                  placeholder={t('session.enterSessionDescription')}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
                  rows={3}
                />
              </div>
            </div>

            <div className="flex space-x-3">
              <button
                onClick={() => {
                  setShowCreateModal(false);
                  setNewSessionName("");
                  setNewSessionDescription("");
                }}
                className="flex-1 px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors font-medium"
              >
                {t('common.cancel')}
              </button>
              <button
                onClick={handleCreateSession}
                disabled={!newSessionName.trim() || loading}
                className="flex-1 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? t('session.creating') : t('common.create')}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Edit Session Modal */}
      {showEditModal && (
        <div className="fixed inset-0 bg-gray-900/50 flex items-center justify-center z-[200] p-4">
          <div className="bg-white rounded-xl shadow-xl max-w-md w-full p-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">{t('session.editSession')}</h2>

            <div className="space-y-4 mb-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  {t('session.sessionName')} *
                </label>
                <input
                  type="text"
                  value={newSessionName}
                  onChange={(e) => setNewSessionName(e.target.value)}
                  placeholder={t('session.enterSessionName')}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                  autoFocus
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  {t('project.projectDescription')}
                </label>
                <textarea
                  value={newSessionDescription}
                  onChange={(e) => setNewSessionDescription(e.target.value)}
                  placeholder={t('session.enterSessionDescription')}
                  className="w-full px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
                  rows={3}
                />
              </div>
            </div>

            <div className="flex space-x-3">
              <button
                onClick={() => {
                  setShowEditModal(false);
                  setEditingSession(null);
                  setNewSessionName("");
                  setNewSessionDescription("");
                }}
                className="flex-1 px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300 transition-colors font-medium"
              >
                {t('common.cancel')}
              </button>
              <button
                onClick={handleUpdateSession}
                disabled={!newSessionName.trim() || loading}
                className="flex-1 px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors font-medium disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? t('session.updating') : t('session.update')}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Delete Confirmation Dialog */}
      {showDeleteDialog && (
        <div className="fixed inset-0 bg-gray-900/50 flex items-center justify-center z-[200] p-4">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4 shadow-xl">
            <h3 className="text-lg font-semibold mb-2">{t('common.confirm')}</h3>
            <p className="text-gray-600 mb-4">
              {t('session.deleteConfirm')}
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
