// API Types

export type BoundingBox = {
  pts_s: number;
  conf: number;
  left: number;
  top: number;
  width: number;
  height: number;
};

export type StatusResponse = {
  armed: boolean;
  tilt: number | null;
  pan: number | null;
  preview: string | null;
  bbox: BoundingBox | null;
};

export type Recording = {
  id: string;
  filename: string;
  createdAt: string;
  durationSeconds: number;
  sizeBytes: number;
};

export type RecordingStatusResponse = {
  recording: Recording;
  status: "recording" | "stopped";
};

export type RecordingListResponse = {
  recordings: Recording[];
};

export type RecordingResponse = {
  recording: Recording;
};

export type ApiResponse<T = Record<string, unknown>> = T;

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

/**
 * API Client for communicating with the Flask backend
 */
export class ApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = "") {
    this.baseUrl = baseUrl;
  }

  /**
   * Automatically creates an ApiClient by trying different base URLs in order.
   * Tests each URL by calling getStatus() and returns the first working instance.
   * @returns Promise resolving to an ApiClient instance with a working base URL
   * @throws Error if none of the base URLs are accessible
   */
  static async createAutomatic(): Promise<ApiClient> {
    const baseUrls = ["", "http://localhost:5000", "http://100.115.14.44"];

    for (const baseUrl of baseUrls) {
      const client = new ApiClient(baseUrl);

      try {
        await client.getStatus();
        console.log(`Connected to API at ${baseUrl}`);

        return client;
      } catch (error) {
        // Continue to next URL if this one fails
        continue;
      }
    }

    throw new Error(
      "Failed to connect to API. Tried base URLs: " + baseUrls.join(", "),
    );
  }

  previewUrl(): string {
    return `${this.baseUrl}/preview`;
  }

  downloadRecordingUrl(recordingId: string): string {
    return `${this.baseUrl}/api/recordings/${recordingId}/download`;
  }

  private async requestJson<T>(
    method: "GET" | "POST" | "PATCH" | "DELETE",
    endpoint: string,
    body?: Record<string, unknown>,
  ): Promise<T> {
    const url = `${this.baseUrl}${endpoint}`;
    const response = await fetch(url, {
      method,
      headers: {
        "Content-Type": "application/json",
      },
      body: body ? JSON.stringify(body) : undefined,
    });

    const text = await response.text();
    const data = text ? (JSON.parse(text) as { error?: string }) : null;

    if (!response.ok) {
      const message = data?.error || response.statusText || "Request failed";

      throw new ApiError(response.status, message);
    }

    if (!text) {
      return {} as T;
    }

    return data as T;
  }

  /**
   * Makes a POST request to the API
   */
  private async post<T>(
    endpoint: string,
    body?: Record<string, unknown>,
  ): Promise<T> {
    return this.requestJson<T>("POST", endpoint, body);
  }

  private async get<T>(endpoint: string): Promise<T> {
    return this.requestJson<T>("GET", endpoint);
  }

  private async patch<T>(
    endpoint: string,
    body?: Record<string, unknown>,
  ): Promise<T> {
    return this.requestJson<T>("PATCH", endpoint, body);
  }

  private async delete<T>(endpoint: string): Promise<T> {
    return this.requestJson<T>("DELETE", endpoint);
  }

  /**
   * Gets the current status from the backend
   * @returns Promise resolving to the status object
   */
  async getStatus(): Promise<ApiResponse<StatusResponse>> {
    return this.post<ApiResponse<StatusResponse>>("/api/status");
  }

  /**
   * Sends a manual move command to the backend
   * @param direction - The direction to move
   * @returns Promise resolving to an empty response
   */
  async manualMove(
    direction: "up" | "down" | "left" | "right",
  ): Promise<ApiResponse> {
    const body = { direction };

    return this.post<ApiResponse>("/api/manual_move", body);
  }

  /**
   * Sends a manual move to command to the backend
   * @param tilt - The tilt angle to move to
   * @param pan - The pan angle to move to
   * @returns Promise resolving to an empty response
   */
  async manualMoveTo(tilt: number, pan: number): Promise<ApiResponse> {
    const body = { tilt, pan };

    return this.post<ApiResponse>("/api/manual_move_to", body);
  }

  /**
   * Arms the system
   * @returns Promise resolving to an empty response
   */
  async arm(): Promise<ApiResponse> {
    return this.post<ApiResponse>("/api/arm");
  }

  /**
   * Disarms the system
   * @returns Promise resolving to an empty response
   */
  async disarm(): Promise<ApiResponse> {
    return this.post<ApiResponse>("/api/disarm");
  }

  async startRecording(): Promise<RecordingStatusResponse> {
    return this.post<RecordingStatusResponse>("/api/recordings/start");
  }

  async stopRecording(): Promise<RecordingStatusResponse> {
    return this.post<RecordingStatusResponse>("/api/recordings/stop");
  }

  async listRecordings(): Promise<RecordingListResponse> {
    return this.get<RecordingListResponse>("/api/recordings");
  }

  async getRecording(recordingId: string): Promise<RecordingResponse> {
    return this.get<RecordingResponse>(`/api/recordings/${recordingId}`);
  }

  async renameRecording(
    recordingId: string,
    filename: string,
  ): Promise<ApiResponse> {
    return this.patch<ApiResponse>(`/api/recordings/${recordingId}`, {
      filename,
    });
  }

  async deleteRecording(recordingId: string): Promise<ApiResponse> {
    return this.delete<ApiResponse>(`/api/recordings/${recordingId}`);
  }
}

// Export a default instance
export const apiClient = new ApiClient();
