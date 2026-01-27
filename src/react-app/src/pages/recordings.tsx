import type { Recording, ApiClient } from '@/network/api'

import { useEffect, useState } from 'react'
import { Button } from '@heroui/button'

import DefaultLayout from '@/layouts/default'
import { useRocam } from '@/network/rocamProvider'

export default function RecordingsPage() {
  const { apiClient } = useRocam()

  const [recordings, setRecordings] = useState<Recording[]>([])

  async function loadRecordings() {
    if (!apiClient) return

    try {
      const data = await apiClient.listRecordings()

      setRecordings(data.recordings)
    } catch (e) {
      console.error('Failed to load recordings:', e)
    }
  }

  useEffect(() => {
    if (apiClient) {
      loadRecordings()
    }
  }, [apiClient])

  const handleRename = async (id: string, newName: string) => {
    if (!apiClient) return
    try {
      const res = await apiClient.renameRecording(id, newName)
      const updated = (res as any)?.recording as Recording | undefined

      if (updated) {
        setRecordings((cur) => cur.map((x) => (x.id === id ? updated : x)))
      } else {
        await loadRecordings()
      }
    } catch (e) {
      console.error('Failed to rename recording:', e)
      throw e
    }
  }

  const handleDelete = async (r: Recording) => {
    if (!apiClient) return
    if (!confirm(`Delete "${r.filename}"? This cannot be undone.`)) return

    try {
      await apiClient.deleteRecording(r.id)
      setRecordings((cur) => cur.filter((x) => x.id !== r.id))
    } catch (e) {
      console.error('Failed to delete recording:', e)
      throw e
    }
  }

  return (
    <DefaultLayout>
      <section className="flex flex-col">
        <div className="divide-y divide-gray-200 px-4">
          {recordings.length === 0 ? (
            <p className="text-sm text-gray-500">
              No recordings yet. Start one from the Control page.
            </p>
          ) : (
            recordings.map((r) => (
              <RecordingItem
                key={r.id}
                apiClient={apiClient}
                recording={r}
                onDelete={handleDelete}
                onRename={handleRename}
              />
            ))
          )}
        </div>
      </section>
    </DefaultLayout>
  )
}

/**
 * SUB-COMPONENTS
 */

interface RecordingItemProps {
  recording: Recording
  apiClient: ApiClient | null
  onRename: (id: string, newName: string) => Promise<void>
  onDelete: (r: Recording) => Promise<void>
}

function RecordingItem({
  recording: r,
  apiClient,
  onRename,
  onDelete,
}: RecordingItemProps) {
  const [isEditing, setIsEditing] = useState(false)
  const [filenameDraft, setFilenameDraft] = useState(r.filename)
  const [isSaving, setIsSaving] = useState(false)
  const [isDeleting, setIsDeleting] = useState(false)

  const handleSave = async () => {
    if (!filenameDraft.trim() || isSaving) return
    setIsSaving(true)
    try {
      await onRename(r.id, filenameDraft.trim())
      setIsEditing(false)
    } catch {
      // Error handled by parent
    } finally {
      setIsSaving(false)
    }
  }

  const handleDelete = async () => {
    if (isDeleting) return
    setIsDeleting(true)
    try {
      await onDelete(r)
    } catch {
      // Error handled by parent
    } finally {
      setIsDeleting(false)
    }
  }

  return (
    <div className="bg-white py-4">
      <div className="flex items-center justify-between gap-3">
        <div className="flex-1 min-w-0">
          {isEditing ? (
            <input
              className="w-full rounded border border-gray-200 px-2 py-1 text-sm"
              value={filenameDraft}
              onChange={(ev) => setFilenameDraft(ev.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleSave()
                if (e.key === 'Escape') {
                  setIsEditing(false)
                  setFilenameDraft(r.filename)
                }
              }}
              autoFocus
            />
          ) : (
            <p className="font-medium truncate">{r.filename}</p>
          )}

          <p className="text-xs text-gray-500 mt-1">
            Created: {formatDate(r.createdAt)} • Duration:{' '}
            {formatDuration(r.durationSeconds)} • Size:{' '}
            {formatBytes(r.sizeBytes)}
          </p>
        </div>

        <div className="flex items-center gap-2 flex-wrap justify-end">
          {apiClient ? (
            <a
              className="inline-flex"
              href={apiClient.downloadRecordingUrl(r.id)}
              rel="noreferrer"
              target="_blank"
            >
              <Button radius="sm" size="sm" variant="bordered">
                Download
              </Button>
            </a>
          ) : (
            <Button isDisabled radius="sm" size="sm" variant="bordered">
              Download
            </Button>
          )}

          {isEditing ? (
            <>
              <Button
                isDisabled={isSaving}
                radius="sm"
                size="sm"
                variant="solid"
                onPress={handleSave}
              >
                {isSaving ? 'Saving...' : 'Save'}
              </Button>
              <Button
                radius="sm"
                size="sm"
                variant="bordered"
                onPress={() => {
                  setIsEditing(false)
                  setFilenameDraft(r.filename)
                }}
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
                onPress={() => setIsEditing(true)}
              >
                Rename
              </Button>
              <Button
                color="danger"
                isDisabled={isDeleting}
                radius="sm"
                size="sm"
                variant="bordered"
                onPress={handleDelete}
              >
                {isDeleting ? 'Deleting...' : 'Delete'}
              </Button>
            </>
          )}
        </div>
      </div>

      <div className="mt-2 text-xs text-gray-500 font-mono truncate">
        ID: {r.id}
      </div>
    </div>
  )
}

/**
 * UTILS
 */

function formatDate(iso: string) {
  const d = new Date(iso)
  return isNaN(d.getTime()) ? iso : d.toLocaleString()
}

function formatDuration(seconds: number) {
  if (!Number.isFinite(seconds) || seconds < 0) return '-'
  if (seconds < 60) return `${seconds.toFixed(1)}s`
  const mins = Math.floor(seconds / 60)
  const secs = Math.floor(seconds % 60)
  return `${mins}m ${secs}s`
}

function formatBytes(bytes: number) {
  if (!Number.isFinite(bytes) || bytes < 0) return '-'
  const units = ['B', 'KB', 'MB', 'GB']
  let b = bytes
  let i = 0
  while (b >= 1024 && i < units.length - 1) {
    b /= 1024
    i++
  }
  return `${b.toFixed(i === 0 ? 0 : 1)} ${units[i]}`
}
