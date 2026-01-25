import { useEffect, useMemo, useState } from "react";
import { Button } from "@heroui/button";

import DefaultLayout from "@/layouts/default";
import { useRocam } from "@/network/rocamProvider";
import type { Recording } from "@/network/api";

export default function RecordingsPage() {
  const { apiClient } = useRocam();

  const [recordings, setRecordings] = useState<Recording[]>([]);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const [editingId, setEditingId] = useState<string | null>(null);
  const [filenameDraft, setFilenameDraft] = useState("");
  const [savingId, setSavingId] = useState<string | null>(null);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const [playing, setPlaying] = useState<Recording | null>(null);

  const totalSizeBytes = useMemo(
    () => recordings.reduce((sum, r) => sum + (r.sizeBytes ?? 0), 0),
    [recordings],
  );

  async function loadRecordings() {
    if (!apiClient) return;
    try {
      const data = await apiClient.listRecordings();
      setRecordings(data.recordings);
      setErrorMessage(null);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Failed to load recordings.";
      setErrorMessage(msg);
    }
  }

  useEffect(() => {
    if (!apiClient) return;
    loadRecordings();
  }, [apiClient]);

  const startEditing = (r: Recording) => {
    setEditingId(r.id);
    setFilenameDraft(r.filename);
  };

  const cancelEditing = () => {
    setEditingId(null);
    setFilenameDraft("");
  };

  const handleSave = async (r: Recording) => {
    if (!apiClient || savingId || !filenameDraft.trim()) return;

    setSavingId(r.id);
    try {
      const res = await apiClient.renameRecording(r.id, filenameDraft.trim());

      // Backend may return {} or { recording: Recording }
      const updated = (res as any)?.recording as Recording | undefined;

      if (updated) {
        setRecordings((cur) => cur.map((x) => (x.id === r.id ? updated : x)));
      } else {
        await loadRecordings();
      }

      cancelEditing();
      setErrorMessage(null);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Failed to rename recording.";
      setErrorMessage(msg);
    } finally {
      setSavingId(null);
    }
  };

  const handleDelete = async (r: Recording) => {
    if (!apiClient || deletingId) return;

    if (!confirm(`Delete "${r.filename}"? This cannot be undone.`)) return;

    setDeletingId(r.id);
    try {
      await apiClient.deleteRecording(r.id);
      setRecordings((cur) => cur.filter((x) => x.id !== r.id));
      setErrorMessage(null);
    } catch (e) {
      const msg = e instanceof Error ? e.message : "Failed to delete recording.";
      setErrorMessage(msg);
    } finally {
      setDeletingId(null);
    }
  };

  const handlePlay = (r: Recording) => setPlaying(r);
  const handleClosePlayer = () => setPlaying(null);

  return (
    <DefaultLayout>
      <section className="flex flex-col gap-6 py-8 md:py-10">
        <header className="flex flex-col gap-2">
          <h1 className="text-3xl font-semibold">Recordings</h1>
          <p className="text-sm text-gray-500">
            Manage videos: play, download, rename, and delete recordings.
          </p>
        </header>

        {errorMessage && (
          <div className="rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">
            {errorMessage}
          </div>
        )}

        <div className="bg-gray-100 rounded-lg p-4">
          <div className="flex items-center justify-between mb-3">
            <div>
              <h2 className="text-lg font-medium">All recordings</h2>
              <p className="text-xs text-gray-500">
                {recordings.length} total • {formatBytes(totalSizeBytes)}
              </p>
            </div>

            <Button
              radius="sm"
              variant="bordered"
              isDisabled={!apiClient}
              onPress={loadRecordings}
            >
              Refresh
            </Button>
          </div>

          <div className="space-y-3">
            {recordings.length === 0 ? (
              <p className="text-sm text-gray-500">
                No recordings yet. Start one from the Control page.
              </p>
            ) : (
              recordings.map((r) => (
                <div key={r.id} className="rounded-md bg-white p-3 shadow-sm">
                  <div className="flex items-center justify-between gap-3">
                    <div className="flex-1 min-w-0">
                      {editingId === r.id ? (
                        <input
                          className="w-full rounded border border-gray-200 px-2 py-1 text-sm"
                          value={filenameDraft}
                          onChange={(ev) => setFilenameDraft(ev.target.value)}
                        />
                      ) : (
                        <p className="font-medium truncate">{r.filename}</p>
                      )}

                      <p className="text-xs text-gray-500 mt-1">
                        Created: {formatDate(r.createdAt)} • Duration:{" "}
                        {formatDuration(r.durationSeconds)} • Size:{" "}
                        {formatBytes(r.sizeBytes)}
                      </p>
                    </div>

                    <div className="flex items-center gap-2 flex-wrap justify-end">
                      <Button
                        radius="sm"
                        size="sm"
                        variant="solid"
                        isDisabled={!apiClient}
                        onPress={() => handlePlay(r)}
                      >
                        Play
                      </Button>

                      {apiClient ? (
                        <a
                          className="inline-flex"
                          href={apiClient.downloadRecordingUrl(r.id)}
                          target="_blank"
                          rel="noreferrer"
                        >
                          <Button radius="sm" size="sm" variant="bordered">
                            Download
                          </Button>
                        </a>
                      ) : (
                        <Button radius="sm" size="sm" variant="bordered" isDisabled>
                          Download
                        </Button>
                      )}

                      {editingId === r.id ? (
                        <>
                          <Button
                            radius="sm"
                            size="sm"
                            variant="solid"
                            isDisabled={savingId === r.id}
                            onPress={() => handleSave(r)}
                          >
                            {savingId === r.id ? "Saving..." : "Save"}
                          </Button>
                          <Button
                            radius="sm"
                            size="sm"
                            variant="bordered"
                            onPress={cancelEditing}
                          >
                            Cancel
                          </Button>
                        </>
                      ) : (
                        <>
                          <Button
                            radius="sm"
                            size="sm"
                            variant="bordered"
                            onPress={() => startEditing(r)}
                          >
                            Rename
                          </Button>
                          <Button
                            color="danger"
                            radius="sm"
                            size="sm"
                            variant="bordered"
                            isDisabled={deletingId === r.id}
                            onPress={() => handleDelete(r)}
                          >
                            {deletingId === r.id ? "Deleting..." : "Delete"}
                          </Button>
                        </>
                      )}
                    </div>
                  </div>

                  <div className="mt-2 text-xs text-gray-500 font-mono truncate">
                    ID: {r.id}
                  </div>
                </div>
              ))
            )}
          </div>
        </div>

        {/* PLAYER MODAL */}
        {playing && apiClient && (
          <div
            className="fixed inset-0 bg-black/50 flex items-center justify-center p-4"
            role="dialog"
            aria-modal="true"
            onClick={handleClosePlayer}
          >
            <div
              className="bg-white rounded-lg w-full max-w-3xl p-4"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between mb-3">
                <div className="min-w-0">
                  <p className="font-medium truncate">{playing.filename}</p>
                  <p className="text-xs text-gray-500">
                    {formatDate(playing.createdAt)}
                  </p>
                </div>
                <Button radius="sm" variant="bordered" onPress={handleClosePlayer}>
                  Close
                </Button>
              </div>

              <video
                className="w-full rounded"
                controls
                src={apiClient.downloadRecordingUrl(playing.id)}
              />
            </div>
          </div>
        )}
      </section>
    </DefaultLayout>
  );
}

function formatDate(iso: string) {
  const d = new Date(iso);
  return isNaN(d.getTime()) ? iso : d.toLocaleString();
}

function formatDuration(seconds: number) {
  if (!Number.isFinite(seconds) || seconds < 0) return "-";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const mins = Math.floor(seconds / 60);
  const secs = Math.floor(seconds % 60);
  return `${mins}m ${secs}s`;
}

function formatBytes(bytes: number) {
  if (!Number.isFinite(bytes) || bytes < 0) return "-";
  const units = ["B", "KB", "MB", "GB"];
  let b = bytes;
  let i = 0;
  while (b >= 1024 && i < units.length - 1) {
    b /= 1024;
    i++;
  }
  return `${b.toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}
