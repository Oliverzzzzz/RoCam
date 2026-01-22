const express = require('express');
const app = express();
app.use(express.json());

// --- Mock System State ---
let state = {
    armed: false,
    tilt: 0,
    pan: 0,
    isRecording: false,
    // Base64-encoded 1x1 black pixel as a placeholder for the preview frame
    preview: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
    bbox: {
        pts_s: 0.0,
        conf: 0.98,
        left: 100,
        top: 100,
        width: 50,
        height: 50
    },
    recordings: [
        {
            id: "rec_001",
            filename: "training_data_01.mp4",
            createdAt: new Date().toISOString(),
            durationSeconds: 120.5,
            sizeBytes: 104857600 // 100MB
        }
    ]
};

// --- GET /api/status ---
app.post('/api/status', (req, res) => {
    res.json({
        armed: state.armed,
        tilt: state.tilt,
        pan: state.pan,
        preview: state.preview,
        bbox: state.armed ? state.bbox : null
    });
});

// --- Gimbal Control ---
app.post('/api/manual_move', (req, res) => {
    const { direction } = req.body;
    const step = 5;
    if (direction === 'up') state.tilt += step;
    if (direction === 'down') state.tilt -= step;
    if (direction === 'left') state.pan -= step;
    if (direction === 'right') state.pan += step;
    res.json({});
});

app.post('/api/manual_move_to', (req, res) => {
    state.tilt = req.body.tilt;
    state.pan = req.body.pan;
    res.json({});
});

app.post('/api/gimbal/home', (req, res) => {
    state.tilt = 0;
    state.pan = 0;
    res.json({});
});

// --- Tracking Arming ---
app.post('/api/arm', (req, res) => {
    state.armed = true;
    res.json({});
});

app.post('/api/disarm', (req, res) => {
    state.armed = false;
    res.json({});
});

// --- Recording Management ---
app.post('/api/recordings/start', (req, res) => {
    state.isRecording = true;
    const newRecording = {
        id: `rec_${Date.now()}`,
        filename: `video_${Date.now()}.mp4`,
        createdAt: new Date().toISOString(),
        durationSeconds: 0,
        sizeBytes: 0
    };
    state.recordings.push(newRecording);
    res.json({ recording: newRecording, status: "recording" });
});

app.post('/api/recordings/stop', (req, res) => {
    state.isRecording = false;
    const lastRec = state.recordings[state.recordings.length - 1];
    res.json({ recording: lastRec, status: "stopped" });
});

app.get('/api/recordings', (req, res) => {
    res.json({ recordings: state.recordings });
});

app.get('/api/recordings/:recordingId', (req, res) => {
    const rec = state.recordings.find(r => r.id === req.params.recordingId);
    rec ? res.json({ recording: rec }) : res.status(404).json({ error: "Not found" });
});

app.delete('/api/recordings/:recordingId', (req, res) => {
    state.recordings = state.recordings.filter(r => r.id !== req.params.recordingId);
    res.json({});
});

// --- Binary Download ---
app.get('/api/recordings/:recordingId/download', (req, res) => {
    const rec = state.recordings.find(r => r.id === req.params.recordingId);
    if (!rec) return res.status(404).send("File not found");

    res.setHeader('Content-Type', 'video/mp4');
    res.setHeader('Content-Disposition', `attachment; filename="${rec.filename}"`);
    // Sending a fake 10-byte buffer to simulate binary video data
    res.send(Buffer.alloc(10, 'AB')); 
});

const PORT = 5000;
app.listen(PORT, () => {
    console.log(`Vision Guided Tracker Mock API running at http://localhost:${PORT}`);
});