import type { Recording } from "@/network/api";

export const MOCK_RECORDINGS: Recording[] = [
  {
    id: "mock-1",
    filename: "sample_001.mp4",
    createdAt: new Date().toISOString(),
    durationSeconds: 12.3,
    sizeBytes: 15_000_000,
  },
  {
    id: "mock-2",
    filename: "sample_002.mp4",
    createdAt: new Date(Date.now() - 60 * 60 * 1000).toISOString(),
    durationSeconds: 45.0,
    sizeBytes: 52_000_000,
  },
];
