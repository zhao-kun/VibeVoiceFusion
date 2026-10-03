# VibeVoice API Documentation

This document provides complete API documentation for VibeVoice backend services.

## Base URL

```
http://localhost:9527/api/v1
```

## Table of Contents

1. [Projects API](#projects-api)
2. [Speakers API](#speakers-api)
3. [Preset Voices API](#preset-voices-api)
4. [Dialog Sessions API](#dialog-sessions-api)
5. [Tasks API (Unified)](#tasks-api-unified)
6. [Generation API](#generation-api)
7. [Quick Generate API](#quick-generate-api)
8. [Transcription API](#transcription-api)
9. [OpenAI-Compatible TTS API](#openai-compatible-tts-api) (speech, transcriptions and [realtime transcription over WebSocket](#3-realtime-transcription-websocket))
10. [Dataset Management API](#dataset-management-api)
11. [Training Management API](#training-management-api)

---

# Projects API

## Overview

The Projects API provides endpoints for managing projects. Each project is a workspace containing speakers, dialog sessions, datasets, and training jobs.

## Endpoints

### 1. List Projects

**GET** `/api/v1/projects`

Get all projects.

**Response (200 OK):**
```json
{
  "projects": [
    {
      "id": "my-project",
      "name": "My Project",
      "created_at": "2025-12-02T10:00:00Z"
    }
  ]
}
```

### 2. Create Project

**POST** `/api/v1/projects`

Create a new project.

**Request Body:**
```json
{
  "name": "My New Project"
}
```

**Response (201 Created):**
```json
{
  "id": "generated-project-id",
  "name": "My New Project",
  "created_at": "2025-12-02T10:00:00Z"
}
```

### 3. Get Project

**GET** `/api/v1/projects/{project_id}`

Get project details.

**Response (200 OK):**
```json
{
  "id": "my-project",
  "name": "My Project",
  "created_at": "2025-12-02T10:00:00Z"
}
```

### 4. Update Project

**PUT** `/api/v1/projects/{project_id}`

Update project details.

**Request Body:**
```json
{
  "name": "Updated Project Name"
}
```

**Response (200 OK):**
```json
{
  "id": "my-project",
  "name": "Updated Project Name",
  "created_at": "2025-12-02T10:00:00Z"
}
```

### 5. Delete Project

**DELETE** `/api/v1/projects/{project_id}`

Delete a project and all its data.

**Response (200 OK):**
```json
{
  "message": "Project deleted successfully"
}
```

---

# Speakers API

## Overview

The Speakers API manages speaker voice samples within a project. Each speaker has a unique ID and one or more voice samples.

## Endpoints

### 1. List Speakers

**GET** `/api/v1/projects/{project_id}/speakers`

Get all speakers for a project.

**Response (200 OK):**
```json
{
  "speakers": [
    {
      "id": "speaker-1-id",
      "name": "Speaker 1",
      "voice_file": "uuid.wav",
      "created_at": "2025-12-02T10:00:00Z"
    }
  ]
}
```

### 2. Create Speaker

**POST** `/api/v1/projects/{project_id}/speakers`

Upload a new speaker voice sample.

**Form Data:**
- `name` (required) - Speaker name
- `voice_file` (required) - Audio file (WAV, MP3, M4A, FLAC, WEBM)

**Response (201 Created):**
```json
{
  "id": "speaker-1-id",
  "name": "Speaker 1",
  "voice_file": "uuid.wav",
  "created_at": "2025-12-02T10:00:00Z"
}
```

### 3. Update Speaker

**PUT** `/api/v1/projects/{project_id}/speakers/{speaker_id}`

Update speaker details or voice sample.

**Form Data:**
- `name` (optional) - New speaker name
- `voice_file` (optional) - New audio file

**Response (200 OK):**
```json
{
  "id": "speaker-1-id",
  "name": "Updated Speaker Name",
  "voice_file": "new-uuid.wav",
  "created_at": "2025-12-02T10:00:00Z"
}
```

### 4. Delete Speaker

**DELETE** `/api/v1/projects/{project_id}/speakers/{speaker_id}`

Delete a speaker and its voice sample.

**Response (200 OK):**
```json
{
  "message": "Speaker deleted successfully"
}
```

### 5. Get Speaker Voice File

**GET** `/api/v1/projects/{project_id}/speakers/{speaker_id}/voice`

Download the speaker's voice file.

**Response (200 OK):**
- Content-Type: `audio/wav` (or appropriate audio type)
- Binary audio data

### 6. Create Speaker from Preset

**POST** `/api/v1/projects/{project_id}/speakers/from-preset`

Create a new speaker using a preset voice file.

**Request Body:**
```json
{
  "preset_id": "a1b2c3d4e5f6",
  "description": "Main character voice"
}
```

Or using legacy filename (for backwards compatibility):
```json
{
  "preset_filename": "a1b2c3d4e5f6.wav",
  "description": "Main character voice"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `preset_id` | string | * | Preset voice ID (preferred) |
| `preset_filename` | string | * | Preset voice filename (legacy) |
| `description` | string | No | Speaker description |

*Either `preset_id` or `preset_filename` is required.

**Response (201 Created):**
```json
{
  "speaker_id": "Speaker 1",
  "description": "Main character voice",
  "voice_filename": "a1b2c3d4e5f6.wav",
  "created_at": "2025-12-19T10:00:00Z",
  "updated_at": "2025-12-19T10:00:00Z"
}
```

**Error Responses:**
- `400 Bad Request`: Missing both preset_id and preset_filename
- `404 Not Found`: Project or preset voice not found

---

# Preset Voices API

## Overview

The Preset Voices API provides full management of preset voice samples. Preset voices are global (not project-scoped) and can be used across all projects to quickly create speakers.

**Storage Location:** `backend/preset_voice/`

## Data Model

### PresetVoice

```json
{
  "id": "a1b2c3d4e5f6",
  "name": "Alice",
  "language": "en",
  "has_bgm": false,
  "voice_filename": "a1b2c3d4e5f6.wav",
  "created_at": "2025-12-21T10:00:00Z",
  "updated_at": "2025-12-21T10:00:00Z"
}
```

## Endpoints

### 1. List Preset Voices

**GET** `/api/v1/preset-voices`

Get all preset voices with optional filtering and pagination.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `language` | string | Filter by language code (`en`, `zh`) |
| `has_bgm` | boolean | Filter by BGM presence (`true`, `false`) |
| `offset` | integer | Number of items to skip (default: 0) |
| `limit` | integer | Maximum items per page (optional) |

**Response (200 OK):**
```json
{
  "presets": [
    {
      "id": "a1b2c3d4e5f6",
      "name": "Alice",
      "language": "en",
      "has_bgm": false,
      "voice_filename": "a1b2c3d4e5f6.wav",
      "created_at": "2025-12-21T10:00:00Z",
      "updated_at": "2025-12-21T10:00:00Z"
    }
  ],
  "count": 1,
  "total": 10,
  "offset": 0,
  "limit": 10
}
```

### 2. Create Preset Voice

**POST** `/api/v1/preset-voices`

Add a new preset voice.

**Form Data:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Voice name |
| `language` | string | Yes | Language code (`en` or `zh`) |
| `has_bgm` | string | Yes | `true` or `false` |
| `voice_file` | file | Yes | Audio file (WAV, MP3, M4A, FLAC, WEBM) |

**Response (201 Created):**
```json
{
  "id": "a1b2c3d4e5f6",
  "name": "Alice",
  "language": "en",
  "has_bgm": false,
  "voice_filename": "a1b2c3d4e5f6.wav",
  "created_at": "2025-12-21T10:00:00Z",
  "updated_at": "2025-12-21T10:00:00Z"
}
```

**Error Responses:**
- `400 Bad Request`: Missing required fields or invalid file type

### 3. Get Preset Voice

**GET** `/api/v1/preset-voices/{preset_id}`

Get a specific preset voice by ID.

**Response (200 OK):**
```json
{
  "id": "a1b2c3d4e5f6",
  "name": "Alice",
  "language": "en",
  "has_bgm": false,
  "voice_filename": "a1b2c3d4e5f6.wav",
  "created_at": "2025-12-21T10:00:00Z",
  "updated_at": "2025-12-21T10:00:00Z"
}
```

**Error Responses:**
- `404 Not Found`: Preset voice not found

### 4. Update Preset Voice

**PUT** `/api/v1/preset-voices/{preset_id}`

Update preset voice metadata.

**Request Body (JSON):**
```json
{
  "name": "Updated Name",
  "language": "zh",
  "has_bgm": true
}
```

All fields are optional. Only provided fields will be updated.

**Response (200 OK):**
```json
{
  "id": "a1b2c3d4e5f6",
  "name": "Updated Name",
  "language": "zh",
  "has_bgm": true,
  "voice_filename": "a1b2c3d4e5f6.wav",
  "created_at": "2025-12-21T10:00:00Z",
  "updated_at": "2025-12-21T10:30:00Z"
}
```

**Error Responses:**
- `400 Bad Request`: Invalid language code
- `404 Not Found`: Preset voice not found

### 5. Delete Preset Voice

**DELETE** `/api/v1/preset-voices/{preset_id}`

Delete a preset voice and its file.

**Response (200 OK):**
```json
{
  "message": "Preset voice deleted successfully",
  "preset_id": "a1b2c3d4e5f6"
}
```

**Error Responses:**
- `404 Not Found`: Preset voice not found

### 6. Batch Delete Preset Voices

**POST** `/api/v1/preset-voices/batch-delete`

Delete multiple preset voices at once.

**Request Body (JSON):**
```json
{
  "preset_ids": ["id1", "id2", "id3"]
}
```

**Response (200 OK):**
```json
{
  "message": "Preset voices deleted successfully",
  "deleted_count": 2,
  "failed_count": 1,
  "deleted_ids": ["id1", "id2"],
  "failed_ids": ["id3"]
}
```

### 7. List Available Languages

**GET** `/api/v1/preset-voices/languages`

Get available languages for preset voices with counts.

**Response (200 OK):**
```json
{
  "languages": [
    {"code": "en", "name": "English", "count": 5},
    {"code": "zh", "name": "Chinese", "count": 3}
  ]
}
```

### 8. Preview Preset Voice

**GET** `/api/v1/preset-voices/{preset_id}/preview`

Stream the preset voice audio file for preview.

**Response (200 OK):**
- Content-Type: `audio/wav`
- Binary audio data

**Error Responses:**
- `404 Not Found`: Preset voice not found

### 9. Preview Preset Voice by Filename (Legacy)

**GET** `/api/v1/preset-voices/by-filename/{filename}/preview`

Stream preset voice audio by filename (legacy endpoint for backwards compatibility).

**Response (200 OK):**
- Content-Type: `audio/wav`
- Binary audio data

**Error Responses:**
- `404 Not Found`: Preset voice not found

---

# Dialog Sessions API

## Overview

The Dialog Sessions API manages dialog scripts within a project. Sessions support two modes:

- **Dialogue Mode** (default): Multi-speaker conversations with "Speaker N: text" format
- **Narration Mode**: Single-speaker plain text for articles, books, narration

## Session Modes

### Dialogue Mode (default)

```
Speaker 1: First line

Speaker 2: Second line

Speaker 1: Can appear multiple times
```

**Critical**: Speaker names must match exactly with uploaded speakers.

### Narration Mode

Plain text without speaker prefixes. All text is spoken by the selected narrator.

```
This is the first paragraph of plain text.

This is the second paragraph. No speaker formatting needed.
```

## Endpoints

### 1. List Sessions

**GET** `/api/v1/projects/{project_id}/sessions`

Get all dialog sessions for a project.

**Response (200 OK):**

```json
{
  "sessions": [
    {
      "session_id": "session-id",
      "name": "My Dialog",
      "description": "Optional description",
      "text_filename": "uuid.txt",
      "mode": "dialogue",
      "narrator_speaker_id": null,
      "created_at": "2025-12-02T10:00:00Z",
      "updated_at": "2025-12-02T10:00:00Z"
    },
    {
      "session_id": "narration-id",
      "name": "My Narration",
      "description": "Article narration",
      "text_filename": "uuid2.txt",
      "mode": "narration",
      "narrator_speaker_id": "Speaker 1",
      "created_at": "2025-12-02T11:00:00Z",
      "updated_at": "2025-12-02T11:00:00Z"
    }
  ]
}
```

### 2. Create Session

**POST** `/api/v1/projects/{project_id}/sessions`

Create a new dialog session.

**Request Body (Dialogue Mode):**

```json
{
  "name": "My Dialog",
  "description": "Optional description",
  "dialog_text": "Speaker 1: Hello\n\nSpeaker 2: Hi there",
  "mode": "dialogue"
}
```

**Request Body (Narration Mode):**

```json
{
  "name": "My Article",
  "description": "Article narration",
  "dialog_text": "This is the first paragraph.\n\nThis is the second paragraph.",
  "mode": "narration",
  "narrator_speaker_id": "Speaker 1"
}
```

**Response (201 Created):**

```json
{
  "session_id": "session-id",
  "name": "My Dialog",
  "description": "Optional description",
  "text_filename": "uuid.txt",
  "mode": "dialogue",
  "narrator_speaker_id": null,
  "created_at": "2025-12-02T10:00:00Z",
  "updated_at": "2025-12-02T10:00:00Z"
}
```

**Validation:**

- `mode`: Must be "dialogue" or "narration" (default: "dialogue")
- `narrator_speaker_id`: Required when mode="narration", must be a valid speaker ID

### 3. Get Session

**GET** `/api/v1/projects/{project_id}/sessions/{session_id}`

Get session details and script content.

**Response (200 OK):**

```json
{
  "session_id": "session-id",
  "name": "My Dialog",
  "description": "Optional description",
  "text_filename": "uuid.txt",
  "mode": "dialogue",
  "narrator_speaker_id": null,
  "created_at": "2025-12-02T10:00:00Z",
  "updated_at": "2025-12-02T10:00:00Z"
}
```

### 4. Update Session

**PUT** `/api/v1/projects/{project_id}/sessions/{session_id}`

Update session metadata, script content, or mode.

**Request Body:**

```json
{
  "name": "Updated Dialog Name",
  "dialog_text": "Speaker 1: Updated script...",
  "mode": "narration",
  "narrator_speaker_id": "Speaker 1"
}
```

**Response (200 OK):**

```json
{
  "session_id": "session-id",
  "name": "Updated Dialog Name",
  "description": "Optional description",
  "text_filename": "uuid.txt",
  "mode": "narration",
  "narrator_speaker_id": "Speaker 1",
  "created_at": "2025-12-02T10:00:00Z",
  "updated_at": "2025-12-02T12:00:00Z"
}
```

### 5. Delete Session

**DELETE** `/api/v1/projects/{project_id}/sessions/{session_id}`

Delete a dialog session.

**Response (200 OK):**

```json
{
  "message": "Session deleted successfully"
}
```

---

# Tasks API (Unified)

## Overview

The Tasks API provides a unified endpoint to check all running tasks (both inference/generation and training). This is used by the navigation task indicator to show when any task is running across all projects.

## Endpoints

### 1. Get Current Task

**GET** `/api/v1/tasks/current`

Get the currently running task (either inference or training) regardless of project.

**Response (200 OK, inference task running):**
```json
{
  "message": "Current inference task retrieved successfully",
  "task": {
    "type": "inference",
    "project_id": "project-id",
    "data": {
      "request_id": "generation-request-id",
      "session_id": "session-id",
      "session_name": "My Dialog",
      "status": "inferencing",
      "details": {
        "current": 45,
        "total_step": 100,
        "percentage": 45.0
      },
      "created_at": "2025-12-02T10:00:00Z"
    }
  }
}
```

**Response (200 OK, training task running):**
```json
{
  "message": "Current training task retrieved successfully",
  "task": {
    "type": "training",
    "project_id": "project-id",
    "data": {
      "task_id": "abc123",
      "job_name": "My LoRA Training",
      "status": "Training",
      "current_step": 150,
      "estimated_total_steps": 1000,
      "current_epoch": 5,
      "total_epochs": 10,
      "current_loss": 0.245
    }
  }
}
```

**Response (200 OK, transcription task running):**
```json
{
  "message": "Current transcription task retrieved successfully",
  "task": {
    "type": "transcription",
    "project_id": null,
    "data": {
      "request_id": "5f0c3c3e9a7b4c8e8f1d2a3b4c5d6e7f",
      "original_filename": "meeting.wav",
      "status": "inferencing",
      "generated_tokens": 128
    }
  }
}
```

`project_id` is `null` for standalone transcriptions and the project ID for project-scoped ones. `data` is the full [Transcription](#transcription) object (abbreviated above).

**Response (200 OK, no active task):**
```json
{
  "message": "No active task at the moment",
  "task": {
    "type": null,
    "project_id": null,
    "data": null
  }
}
```

## Task Types

| Type | Description |
|------|-------------|
| `inference` | Voice generation task is running |
| `training` | LoRA training task is running |
| `transcription` | ASR transcription task is running |
| `null` | No active task |

## Use Case

The unified Tasks API is primarily used for:
- **Navigation task indicator**: Shows a badge when any task is running
- **Cross-project task awareness**: Know if a task is running even when viewing a different project
- **Task type routing**: Navigate to the correct page (generation or training) based on task type

---

# Generation API

## Overview

The Generation API handles voice generation requests. It uses a shared task manager with training (only one task can run at a time).

## Endpoints

### 1. Start Generation

**POST** `/api/v1/projects/{project_id}/generations`

Start a new voice generation task.

**Request Body:**
```json
{
  "dialog_session_id": "session-id",
  "seeds": 42,
  "cfg_scale": 1.3,
  "model_dtype": "float8_e4m3fn",
  "attn_implementation": "sdpa",
  "lora_model_path": "path/to/lora.safetensors",
  "lora_weight": 1.0,
  "offloading": {
    "enabled": true,
    "mode": "preset",
    "preset": "balanced"
  }
}
```

**Response (200 OK):**
```json
{
  "message": "Generation started successfully",
  "request_id": "generation-request-id",
  "generation": {
    "request_id": "abc123",
    "session_id": "session-id",
    "session_name": "My Dialog",
    "status": "pending",
    "created_at": "2025-12-02T10:00:00Z"
  }
}
```

**Response (500 Error - Task Manager Busy):**
```json
{
  "error": "Internal Server Error",
  "message": "A generation task is already running"
}
```

### 2. Get Current Generation (Global)

**GET** `/api/v1/projects/generations/current`

Get current generation task status globally (for any project). Used by the navigation task indicator.

**Response (200 OK, generation in progress):**
```json
{
  "message": "Current generation status retrieved successfully",
  "generation": {
    "request_id": "generation-request-id",
    "project_id": "project-id",
    "session_id": "session-id",
    "session_name": "My Dialog",
    "status": "inferencing",
    "details": {
      "current": 45,
      "total_step": 100,
      "percentage": 45.0
    },
    "created_at": "2025-12-02T10:00:00Z"
  }
}
```

**Response (200 OK, no active generation):**
```json
{
  "message": "No active generation task at the moment",
  "generation": null
}
```

### 3. Get Current Generation (Project-Specific)

**GET** `/api/v1/projects/{project_id}/generations/current`

Get current generation task status for a specific project only. Returns null if the running generation belongs to a different project.

**Response (200 OK, generation in progress for this project):**
```json
{
  "message": "Current generation status retrieved successfully",
  "generation": {
    "request_id": "generation-request-id",
    "project_id": "project-id",
    "session_id": "session-id",
    "session_name": "My Dialog",
    "status": "inferencing",
    "details": {
      "current": 45,
      "total_step": 100,
      "percentage": 45.0
    },
    "created_at": "2025-12-02T10:00:00Z"
  }
}
```

**Response (200 OK, no active generation for this project):**
```json
{
  "message": "No active generation task for this project",
  "generation": null
}
```

### 4. List Generations

**GET** `/api/v1/projects/{project_id}/generations`

Get all generation requests for a project.

**Response (200 OK):**
```json
{
  "generations": [
    {
      "request_id": "generation-request-id",
      "session_id": "session-id",
      "session_name": "My Dialog",
      "status": "completed",
      "output_filename": "uuid.wav",
      "created_at": "2025-12-02T10:00:00Z"
    }
  ],
  "count": 1
}
```

### 5. Get Specific Generation

**GET** `/api/v1/projects/{project_id}/generations/{request_id}`

Get details of a specific generation request.

**Response (200 OK):**
```json
{
  "generation": {
    "request_id": "generation-request-id",
    "session_id": "session-id",
    "session_name": "My Dialog",
    "status": "completed",
    "output_filename": "uuid.wav",
    "created_at": "2025-12-02T10:00:00Z"
  }
}
```

### 6. Download Generated Audio

**GET** `/api/v1/projects/{project_id}/generations/{request_id}/download`

Download or stream the generated audio file.

**Query Parameters:**
- `download` (optional): If `true`, force download as attachment. Otherwise, serve inline for playback.

**Response (200 OK):**
- Content-Type: `audio/wav`
- Binary audio data

**Response (400 Bad Request):**
```json
{
  "error": "Not Found",
  "message": "Generation not found"
}
```

### 7. Download Generation Item Audio (Multi-Generation)

**GET** `/api/v1/projects/{project_id}/generations/{request_id}/items/{item_index}/download`

Download or stream an individual audio file from a multi-generation batch.

**Path Parameters:**
- `project_id`: Project identifier
- `request_id`: Generation request identifier
- `item_index`: Index of the generation item (0-based)

**Query Parameters:**
- `download` (optional): If `true`, force download as attachment. Otherwise, serve inline for playback.

**Response (200 OK):**
- Content-Type: `audio/wav`
- Binary audio data

**Response (404 Not Found):**
```json
{
  "error": "Not Found",
  "message": "Generation item not found"
}
```

**Example:**
```bash
# Stream audio for playback (item 0 from a multi-generation batch)
curl "http://localhost:9527/api/v1/projects/my-project/generations/abc123/items/0/download"

# Force download as attachment
curl -O "http://localhost:9527/api/v1/projects/my-project/generations/abc123/items/2/download?download=true"
```

**Notes:**
- This endpoint is specifically for multi-generation batches where `details.generation_items` contains multiple audio files
- Item index is 0-based (first item is index 0)
- Returns 404 if the generation has no items or the index is out of range
- Each item's audio path is stored in `GenerationItem.audio_path`

---

### 8. Delete Generation

**DELETE** `/api/v1/projects/{project_id}/generations/{request_id}`

Delete a generation request and its output file.

**Response (200 OK):**
```json
{
  "message": "Generation deleted successfully",
  "request_id": "generation-request-id"
}
```

### 9. Batch Delete Generations

**POST** `/api/v1/projects/{project_id}/generations/batch-delete`

Delete multiple generation requests at once.

**Request Body:**
```json
{
  "request_ids": ["id1", "id2", "id3"]
}
```

**Response (200 OK):**
```json
{
  "message": "Generations deleted successfully",
  "deleted_count": 2,
  "failed_count": 1,
  "deleted_ids": ["id1", "id2"],
  "failed_ids": ["id3"]
}
```

## Generation Status Values

| Status | Description |
|--------|-------------|
| `pending` | Task created, waiting to start |
| `preprocessing` | Preparing dialog and speakers |
| `inferencing` | Voice generation in progress |
| `saving_audio` | Saving generated audio to file |
| `completed` | Generation finished successfully |
| `failed` | Generation failed with error |

---

# Quick Generate API

## Overview

The Quick Generate API provides a streamlined voice generation workflow that bypasses project/speaker/session setup. Users can upload a voice sample, provide text, and generate voice immediately. The mode (dialogue or narration) is auto-detected based on text format.

**Storage Location:** `workspace/_quick-generate/`

## Data Models

### QuickGenerate

```json
{
  "request_id": "qg_abc123",
  "status": "completed",
  "detected_mode": "narration",
  "text": "This is sample text for generation.",
  "voice_files": ["voice_abc123.wav", "voice_def456.wav"],
  "voice_file": "voice_abc123.wav",
  "output_files": ["output_abc123_0.wav"],
  "seeds": 42,
  "batch_size": 1,
  "is_multi_generation": false,
  "current_batch_index": 0,
  "percentage": 100.0,
  "model_dtype": "bf16",
  "cfg_scale": 1.3,
  "attn_implementation": "sdpa",
  "offloading": {
    "enabled": false,
    "mode": "preset",
    "preset": "balanced"
  },
  "created_at": "2025-12-02T10:00:00Z",
  "updated_at": "2025-12-02T10:01:00Z",
  "completed_at": "2025-12-02T10:01:00Z",
  "details": {
    "preprocessing_duration": 2.5,
    "offloading_config": {},
    "generation_items": [
      {
        "batch_index": 0,
        "audio_path": "/path/to/output_0.wav",
        "seeds": 42,
        "generation_time": 12.5,
        "audio_duration_seconds": 5.2,
        "real_time_factor": 2.4,
        "current_step": 100,
        "total_steps": 100
      }
    ]
  },
  "error_message": null,
  "text_preview": "This is sample text..."
}
```

### QuickGenerateMode

| Mode | Description |
|------|-------------|
| `dialogue` | Multi-speaker mode, text contains "Speaker X:" prefixes |
| `narration` | Single-speaker mode, plain text without speaker prefixes |

### Mode Auto-Detection

The mode is automatically detected based on text format:
- **Dialogue mode**: If any line matches the pattern `^Speaker\s+\d+\s*:` (e.g., "Speaker 1: Hello")
- **Narration mode**: Plain text without speaker prefixes

In dialogue mode, the uploaded voice samples are used for ALL detected speakers. Up to 4 voice samples can be provided.

## Endpoints

### 1. Start Quick Generation

**POST** `/api/v1/quick-generate`

Start a new quick generation task. Uses multipart form data for voice file upload.

**Form Data:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `voice_files` | file[] | Yes | Voice sample audio files (1-4 files, WAV, MP3, M4A, FLAC, WEBM) |
| `text` | string | Yes | Text to generate (dialogue or narration format) |
| `seeds` | integer | No | Random seed (default: random) |
| `batch_size` | integer | No | Number of generations (1-20, default: 1) |
| `model_dtype` | string | No | Model data type (default: "bf16") |
| `cfg_scale` | float | No | CFG scale (default: 1.3) |
| `offloading_enabled` | boolean | No | Enable layer offloading (default: false) |
| `offloading_preset` | string | No | Offloading preset: "balanced", "aggressive", "extreme" |

**Response (200 OK):**
```json
{
  "message": "Quick generation started successfully",
  "request_id": "qg_abc123",
  "generation": {
    "request_id": "qg_abc123",
    "status": "pending",
    "mode": "narration",
    "text": "This is sample text.",
    "created_at": "2025-12-02T10:00:00Z"
  }
}
```

**Response (400 Bad Request):**
```json
{
  "error": "Bad Request",
  "message": "Voice file is required"
}
```

**Response (409 Conflict):**
```json
{
  "error": "Conflict",
  "message": "A generation task is already running"
}
```

**Example:**
```bash
curl -X POST "http://localhost:9527/api/v1/quick-generate" \
  -F "voice_file=@sample.wav" \
  -F "text=This is the text to generate." \
  -F "seeds=42" \
  -F "batch_size=3"
```

---

### 2. Get Specific Quick Generation

**GET** `/api/v1/quick-generate/{request_id}`

Get details of a specific quick generation request.

**Response (200 OK):**
```json
{
  "generation": {
    "request_id": "qg_abc123",
    "status": "completed",
    "mode": "narration",
    "text": "This is sample text.",
    "voice_file": "voice_abc123.wav",
    "output_file": "output_abc123.wav",
    "created_at": "2025-12-02T10:00:00Z",
    "completed_at": "2025-12-02T10:01:00Z",
    "details": {
      "audio_duration_seconds": 5.2,
      "generation_time": 12.5,
      "real_time_factor": 2.4
    }
  }
}
```

**Response (404 Not Found):**
```json
{
  "error": "Not Found",
  "message": "Quick generation not found"
}
```

---

### 3. Get Current Quick Generation

**GET** `/api/v1/quick-generate/current`

Get the currently running quick generation task with live progress.

**Response (200 OK, generation in progress):**
```json
{
  "message": "Current quick generation retrieved successfully",
  "generation": {
    "request_id": "qg_abc123",
    "status": "inferencing",
    "mode": "narration",
    "text": "This is sample text.",
    "details": {
      "current": 45,
      "total_step": 100,
      "percentage": 45.0
    },
    "created_at": "2025-12-02T10:00:00Z"
  }
}
```

**Response (200 OK, no active generation):**
```json
{
  "message": "No active quick generation task",
  "generation": null
}
```

---

### 4. List Quick Generation History

**GET** `/api/v1/quick-generate/history`

Get quick generation history with optional pagination.

**Query Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `offset` | integer | Number of items to skip (default: 0) |
| `limit` | integer | Maximum items per page (optional) |

**Response (200 OK):**
```json
{
  "generations": [
    {
      "request_id": "qg_abc123",
      "status": "completed",
      "mode": "narration",
      "text": "This is sample text.",
      "created_at": "2025-12-02T10:00:00Z",
      "completed_at": "2025-12-02T10:01:00Z"
    }
  ],
  "count": 1,
  "total": 10,
  "offset": 0,
  "limit": 20
}
```

---

### 5. Download Quick Generation Audio

**GET** `/api/v1/quick-generate/{request_id}/download`

Download or stream the generated audio file.

**Query Parameters:**
- `download` (optional): If `true`, force download as attachment. Otherwise, serve inline for playback.

**Response (200 OK):**
- Content-Type: `audio/wav`
- Binary audio data

**Response (404 Not Found):**
```json
{
  "error": "Not Found",
  "message": "Quick generation not found or audio not ready"
}
```

---

### 6. Download Quick Generation Item Audio (Multi-Generation)

**GET** `/api/v1/quick-generate/{request_id}/items/{item_index}/download`

Download or stream an individual audio file from a multi-generation batch.

**Path Parameters:**
- `request_id`: Quick generation request identifier
- `item_index`: Index of the generation item (0-based)

**Query Parameters:**
- `download` (optional): If `true`, force download as attachment. Otherwise, serve inline for playback.

**Response (200 OK):**
- Content-Type: `audio/wav`
- Binary audio data

**Response (404 Not Found):**
```json
{
  "error": "Not Found",
  "message": "Generation item not found"
}
```

**Example:**
```bash
# Stream audio for playback (item 0 from a multi-generation batch)
curl "http://localhost:9527/api/v1/quick-generate/qg_abc123/items/0/download"

# Force download as attachment
curl -O "http://localhost:9527/api/v1/quick-generate/qg_abc123/items/2/download?download=true"
```

---

### 7. Preview Voice Sample

**GET** `/api/v1/quick-generate/{request_id}/voice-preview`

Stream the first uploaded voice sample for preview (backward compatibility).

**GET** `/api/v1/quick-generate/{request_id}/voice/{voice_index}/preview`

Stream a specific voice sample by index (0-3) for preview.

**Path Parameters:**
| Parameter | Type | Description |
|-----------|------|-------------|
| `request_id` | string | Quick generation request ID |
| `voice_index` | integer | Voice file index (0-3) |

**Response (200 OK):**
- Content-Type: `audio/wav`
- Binary audio data

**Response (404 Not Found):**
```json
{
  "error": "Not Found",
  "message": "Voice file not found"
}
```

---

### 8. Delete Quick Generation

**DELETE** `/api/v1/quick-generate/{request_id}`

Delete a quick generation request and its associated files.

**Response (200 OK):**
```json
{
  "message": "Quick generation deleted successfully",
  "request_id": "qg_abc123"
}
```

**Response (404 Not Found):**
```json
{
  "error": "Not Found",
  "message": "Quick generation not found"
}
```

---

## Quick Generation Status Values

| Status | Description |
|--------|-------------|
| `pending` | Task created, waiting to start |
| `preprocessing` | Preparing voice sample and text |
| `inferencing` | Voice generation in progress |
| `saving_audio` | Saving generated audio to file |
| `completed` | Generation finished successfully |
| `failed` | Generation failed with error |

---

## Workspace Structure

```
workspace/_quick-generate/
├── generations.json       # History of all quick generations
├── voices/                # Uploaded voice samples
│   └── voice_{uuid}.wav
└── outputs/               # Generated audio files
    └── output_{uuid}.wav
```

---

## Multi-Generation Feature

Quick Generate supports batch generation with different random seeds:

- Set `batch_size` > 1 to generate multiple variations
- Each generation uses a different random seed
- Progress shows overall completion and per-item status
- Individual audio files can be downloaded via `/items/{index}/download`

**Data Model:**
```json
{
  "details": {
    "generation_items": [
      {
        "epoch_idx": 0,
        "audio_path": "outputs/item_0_abc123.wav",
        "seeds": 42,
        "generation_time": 4.2,
        "audio_duration_seconds": 5.2,
        "real_time_factor": 1.24
      },
      {
        "epoch_idx": 1,
        "audio_path": "outputs/item_1_abc123.wav",
        "seeds": 123456789,
        "generation_time": 4.1,
        "audio_duration_seconds": 5.2,
        "real_time_factor": 1.27
      }
    ]
  }
}
```

---

# Transcription API

## Overview

The Transcription API transcribes audio with the VibeVoice-ASR model into speaker-labelled, timestamped segments. It runs on the same single-threaded GPU task queue as generation and training, so only one task (of any type) runs at a time.

The same endpoints exist in two scopes that share one engine but keep separate storage:

| Scope | Base path | Storage |
|-------|-----------|---------|
| Standalone (no project) | `/api/v1/transcriptions` | `workspace/_transcriptions/` |
| Project | `/api/v1/projects/{project_id}/transcriptions` | `workspace/{project_id}/transcriptions/` |

A transcription is only visible in the scope it was created in. Project-scoped endpoints return `404` with `errors.project_not_found` when the project does not exist.

**Model location:** `ASR_MODEL_PATH` environment variable (default `./models/VibeVoice-ASR`). The engine prefers a converted single-file checkpoint (`vibevoice_asr_bf16.safetensors` / `vibevoice_asr_float8_e4m3fn.safetensors`) and otherwise loads the original HuggingFace weights from that directory. See [Demo Tools](demo-tools.md) for conversion.

## Data Models

### Transcription

```json
{
  "request_id": "5f0c3c3e9a7b4c8e8f1d2a3b4c5d6e7f",
  "project_id": null,
  "audio_file": "0b8e6c2f4d1a4e3b9c7d5f2a1e3b4c5d.wav",
  "original_filename": "meeting.wav",
  "status": "completed",
  "model_dtype": "bf16",
  "max_new_tokens": 8192,
  "temperature": 0.0,
  "top_p": 1.0,
  "repetition_penalty": 1.0,
  "seeds": 42,
  "context_info": "VibeVoice, Qwen",
  "offloading": null,
  "audio_duration": 63.2,
  "generated_tokens": 412,
  "reach_max_new_tokens": false,
  "raw_text": "[{\"Start\":0.0,\"End\":2.5,\"Speaker\":0,\"Content\":\"Hello everyone.\"}]",
  "segments": [
    {"start_time": 0.0, "end_time": 2.5, "speaker_id": 0, "text": "Hello everyone."}
  ],
  "text_preview": "Hello everyone.",
  "percentage": 100,
  "details": {
    "preprocessing_duration": 0.42,
    "model_load_duration": 18.3,
    "encode_time": 1.2,
    "prefill_time": 0.8,
    "decode_time": 9.6,
    "prompt_tokens": 512,
    "speech_tokens": 474,
    "offloading_config": {}
  },
  "error_message": null,
  "created_at": "2026-10-03T10:00:00",
  "updated_at": "2026-10-03T10:00:31",
  "completed_at": "2026-10-03T10:00:31",
  "source": "file"
}
```

| Field | Description |
|-------|-------------|
| `generated_tokens` | Tokens decoded so far; updated live while `inferencing` |
| `percentage` | `null` while decoding (the final token count is unknown), `100` when completed |
| `reach_max_new_tokens` | `true` if decoding stopped at `max_new_tokens`; the transcript may be truncated |
| `raw_text` | Decoded model output before parsing |
| `segments` | Parsed segments; empty if the model output could not be parsed |
| `source` | `"file"` for uploaded audio, `"live"` for sessions recorded through the [realtime WebSocket](#3-realtime-transcription-websocket). A live record's `original_filename` is `live-YYYYmmdd-HHMMSS.wav`, its `segments` carry absolute session timestamps, and its audio file is written when the session ends |

### Status Values

| Status | Description |
|--------|-------------|
| `pending` | Task queued |
| `preprocessing` | Loading audio and building the prompt |
| `inferencing` | Model loaded, decoding tokens |
| `completed` | Transcription finished |
| `failed` | Transcription failed, see `error_message` |

## Endpoints

Paths below are shown for the standalone scope. For the project scope, replace `/api/v1/transcriptions` with `/api/v1/projects/{project_id}/transcriptions`; the only difference is the history list (see endpoint 4).

### 1. Start Transcription

**POST** `/api/v1/transcriptions`
**POST** `/api/v1/projects/{project_id}/transcriptions`

Upload an audio file and queue a transcription task. Uses multipart form data.

**Form Data:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `audio_file` | file | Yes | Audio file (WAV, MP3, M4A, FLAC, WEBM) |
| `context_info` | string | No | Hotwords, names or topic to guide the model (max 2000 characters) |
| `model_dtype` | string | No | `bf16` (default) or `float8_e4m3fn` |
| `max_new_tokens` | integer | No | 1-32768, default 8192 |
| `temperature` | float | No | 0-2, default 0 (greedy) |
| `top_p` | float | No | 0.01-1, default 1.0 |
| `repetition_penalty` | float | No | 0.5-2, default 1.0 |
| `seeds` | integer | No | Random seed (default: random) |
| `offloading` | string (JSON) | No | Same format as Quick Generate, e.g. `{"enabled": true, "mode": "preset", "preset": "balanced"}` or `{"enabled": true, "mode": "manual", "num_gpu_layers": 20}` (1-28) |

**Response (200 OK):**
```json
{
  "message": "Transcription started successfully",
  "request_id": "5f0c3c3e9a7b4c8e8f1d2a3b4c5d6e7f",
  "project_id": null,
  "status": "pending"
}
```

**Response (400 Bad Request):** missing audio file, unsupported file type, or invalid parameter.

**Response (409 Conflict):** another task is running. The uploaded audio is discarded.
```json
{
  "error": "Conflict",
  "message": "Task manager is busy. Please wait for the current task to complete"
}
```

**Example:**
```bash
curl -X POST "http://localhost:9527/api/v1/transcriptions" \
  -F "audio_file=@meeting.wav" \
  -F "context_info=VibeVoice, Qwen" \
  -F 'offloading={"enabled": true, "mode": "preset", "preset": "balanced"}'
```

---

### 2. Get Transcription

**GET** `/api/v1/transcriptions/{request_id}`
**GET** `/api/v1/projects/{project_id}/transcriptions/{request_id}`

Returns the [Transcription](#transcription) object. While the task is running, the live state is returned (poll every 2 seconds).

**Response (404 Not Found):** transcription does not exist in this scope.

---

### 3. Get Current Transcription

**GET** `/api/v1/transcriptions/current`
**GET** `/api/v1/projects/{project_id}/transcriptions/current`

Returns the running transcription of this scope, or `null`.

**Response (200 OK):**
```json
{
  "message": "Current transcription retrieved successfully",
  "transcription": { "request_id": "...", "status": "inferencing", "generated_tokens": 128 }
}
```

**Response (200 OK, nothing running):**
```json
{
  "message": "No active transcription task",
  "transcription": null
}
```

---

### 4. List Transcriptions

**GET** `/api/v1/transcriptions/history`
**GET** `/api/v1/projects/{project_id}/transcriptions`

Lists transcription summaries, newest first.

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `limit` | integer | 20 | Items per page |
| `offset` | integer | 0 | Items to skip |

**Response (200 OK):**
```json
{
  "transcriptions": [
    {
      "request_id": "5f0c3c3e9a7b4c8e8f1d2a3b4c5d6e7f",
      "project_id": null,
      "status": "completed",
      "original_filename": "meeting.wav",
      "audio_duration": 63.2,
      "text_preview": "Hello everyone.",
      "segment_count": 12,
      "model_dtype": "bf16",
      "created_at": "2026-10-03T10:00:00",
      "completed_at": "2026-10-03T10:00:31",
      "source": "file"
    }
  ],
  "count": 1,
  "total": 1
}
```

---

### 5. Get Source Audio

**GET** `/api/v1/transcriptions/{request_id}/audio`
**GET** `/api/v1/projects/{project_id}/transcriptions/{request_id}/audio`

Streams the uploaded audio file (for playback next to the transcript).

**Response (404 Not Found):** transcription or audio file not found.

---

### 6. Download Transcript

**GET** `/api/v1/transcriptions/{request_id}/download`
**GET** `/api/v1/projects/{project_id}/transcriptions/{request_id}/download`

**Query Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `format` | string | `json` | `json` or `txt` |
| `download` | boolean | `true` | `true` sends `Content-Disposition: attachment` (`{original name}_transcript.{format}`) |

**`txt` format:**
```
[00:00.00 - 00:02.50] Speaker 0: Hello everyone.
[00:02.50 - 00:05.10] Speaker 1: Hi, thanks for joining.
```

**`json` format:**
```json
{
  "request_id": "5f0c3c3e9a7b4c8e8f1d2a3b4c5d6e7f",
  "file": "meeting.wav",
  "audio_duration": 63.2,
  "context_info": "VibeVoice, Qwen",
  "raw_text": "...",
  "segments": [
    {"start_time": 0.0, "end_time": 2.5, "speaker_id": 0, "text": "Hello everyone."}
  ]
}
```

**Response (400 Bad Request):** unsupported `format`.
**Response (404 Not Found):** transcription not found, or not completed yet.

---

### 7. Delete Transcription

**DELETE** `/api/v1/transcriptions/{request_id}`
**DELETE** `/api/v1/projects/{project_id}/transcriptions/{request_id}`

Deletes the record and its uploaded audio file.

**Response (200 OK):**
```json
{
  "message": "Transcription deleted successfully",
  "request_id": "5f0c3c3e9a7b4c8e8f1d2a3b4c5d6e7f"
}
```

**Response (404 Not Found):** transcription not found.
**Response (409 Conflict):** the transcription is still running.

---

### 8. Batch Delete Transcriptions

**POST** `/api/v1/transcriptions/batch-delete`
**POST** `/api/v1/projects/{project_id}/transcriptions/batch-delete`

**Request Body:**
```json
{
  "request_ids": ["id-1", "id-2"]
}
```

**Response (200 OK):** running or unknown IDs are reported in `failed_ids`.
```json
{
  "message": "Transcriptions deleted successfully",
  "deleted_count": 1,
  "failed_count": 1,
  "deleted_ids": ["id-1"],
  "failed_ids": ["id-2"]
}
```

**Response (400 Bad Request):** `request_ids` missing or not a list.

---

## Workspace Structure

```
workspace/
├── _transcriptions/            # Standalone transcriptions
│   ├── history.json
│   └── audio/{uuid}.{ext}
└── {project-id}/
    └── transcriptions/         # Project transcriptions
        ├── history.json
        └── audio/{uuid}.{ext}
```

---

# OpenAI-Compatible TTS API

## Overview

VibeVoice provides an OpenAI-compatible Text-to-Speech endpoint at `POST /v1/audio/speech`, enabling existing OpenAI TTS clients and SDKs to use VibeVoice as a drop-in replacement. A speech-to-text endpoint at `POST /v1/audio/transcriptions` is backed by VibeVoice-ASR (see [Create Transcription](#2-create-transcription)), and live microphone transcription is available over the Realtime WebSocket protocol at `/v1/realtime` (see [Realtime Transcription](#3-realtime-transcription-websocket)).

**Base URL:** `http://localhost:9527/v1` (note: `/v1`, not `/api/v1`)

**Key Differences from OpenAI:**
- Voice selection uses VibeVoice preset voice names (e.g., "Alice", "Bowen") instead of OpenAI voices
- Synchronous blocking: waits for GPU generation to complete before returning
- Returns 503 if the task queue is busy (single-threaded GPU queue)
- `speed` parameter is accepted but ignored

## Authentication

- Accepts `Authorization: Bearer <key>` header
- If `OPENAI_COMPAT_API_KEY` environment variable is set, the key is validated
- If the env var is not set, authentication is skipped (open access)

## Endpoints

### 1. Create Speech

**POST** `/v1/audio/speech`

Generate speech from text using a preset voice. Returns binary audio data directly.

**Request Body (JSON):**
```json
{
  "model": "vibevoice-7b",
  "input": "Hello, this is a test.",
  "voice": "Alice",
  "response_format": "wav",
  "speed": 1.0
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `model` | string | Yes | Model name (see Model Mapping below) |
| `input` | string | Yes | Text to speak (max 4096 characters) |
| `voice` | string | Yes | Preset voice name (case-insensitive) |
| `response_format` | string | No | Output format: `wav` (default), `mp3`, `flac`, `opus`, `aac`, `pcm` |
| `speed` | number | No | Accepted but ignored (not supported by engine) |

**Model Mapping:**

| Model Name | VibeVoice model_dtype | Description |
|------------|----------------------|-------------|
| `vibevoice-7b` | `bf16` | Standard quality, faster |
| `vibevoice-7b-hd` | `float8_e4m3fn` | Higher precision |
| `tts-1` | `bf16` | OpenAI compatibility alias |
| `tts-1-hd` | `float8_e4m3fn` | OpenAI compatibility alias |

**Response (200 OK):**
- Content-Type: `audio/wav`, `audio/mpeg`, `audio/flac`, `audio/opus`, `audio/aac`, or `audio/pcm`
- Binary audio data

**Error Responses (OpenAI-compatible format):**

```json
{
  "error": {
    "message": "Descriptive error message",
    "type": "invalid_request_error",
    "code": "error_code"
  }
}
```

| Status | Type | Code | Description |
|--------|------|------|-------------|
| 400 | `invalid_request_error` | `missing_model` | Missing `model` parameter |
| 400 | `invalid_request_error` | `missing_input` | Missing `input` parameter |
| 400 | `invalid_request_error` | `missing_voice` | Missing `voice` parameter |
| 400 | `invalid_request_error` | `input_too_long` | Input exceeds 4096 characters |
| 400 | `invalid_request_error` | `unsupported_format` | Unsupported response_format |
| 400 | `invalid_request_error` | `model_not_found` | Unknown model name |
| 400 | `invalid_request_error` | `voice_not_found` | Unknown voice name |
| 401 | `authentication_error` | `invalid_api_key` | Invalid API key |
| 503 | `server_error` | - | Task queue is busy |
| 504 | `server_error` | - | Generation timed out |
| 500 | `server_error` | - | Internal error |

**Examples:**

```bash
# Basic usage
curl http://localhost:9527/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"vibevoice-7b","input":"Hello world","voice":"Alice"}' \
  --output speech.wav

# With API key and MP3 format
curl http://localhost:9527/v1/audio/speech \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{"model":"tts-1","input":"Hello world","voice":"Alice","response_format":"mp3"}' \
  --output speech.mp3
```

```python
# Using OpenAI Python SDK
from openai import OpenAI

client = OpenAI(base_url="http://localhost:9527/v1", api_key="unused")
response = client.audio.speech.create(
    model="vibevoice-7b",
    voice="Alice",
    input="Hello, this is a test."
)
response.stream_to_file("output.wav")
```

### 2. Create Transcription

**POST** `/v1/audio/transcriptions`

Transcribe an audio file with VibeVoice-ASR. Synchronous: blocks until the transcription finishes (timeout 1800 seconds). The result is also saved in the standalone transcription history (`/api/v1/transcriptions/history`).

**Request (multipart/form-data):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `file` | file | Yes | Audio file (WAV, MP3, M4A, FLAC, WEBM) |
| `model` | string | Yes | Model name (see Model Mapping below); unknown names fall back to `bf16` |
| `prompt` | string | No | Passed to the model as context info (hotwords, names, topic) |
| `response_format` | string | No | `json` (default), `text`, `verbose_json` |
| `temperature` | number | No | 0-1, default 0 (greedy) |
| `language` | string | No | Accepted but not used by the model; echoed back in `verbose_json` |

Other OpenAI parameters (`timestamp_granularities[]`, `stream`, `include[]`) are not supported and are ignored.

**Model Mapping:**

| Model Name | VibeVoice model_dtype | Description |
|------------|----------------------|-------------|
| `vibevoice-asr` | `bf16` | Standard precision |
| `vibevoice-asr-fp8` | `float8_e4m3fn` | FP8 weights (needs the converted FP8 checkpoint) |
| `whisper-1` | `bf16` | OpenAI compatibility alias |

**Response (200 OK, `json`):**
```json
{
  "text": "Hello everyone.\nHi, thanks for joining."
}
```

**Response (200 OK, `text`):** `text/plain` body with one line per segment.

**Response (200 OK, `verbose_json`):**
```json
{
  "task": "transcribe",
  "language": "unknown",
  "duration": 63.2,
  "text": "Hello everyone.\nHi, thanks for joining.",
  "segments": [
    {"id": 0, "start": 0.0, "end": 2.5, "speaker": 0, "text": "Hello everyone."},
    {"id": 1, "start": 2.5, "end": 5.1, "speaker": 1, "text": "Hi, thanks for joining."}
  ]
}
```

`speaker` is a VibeVoice extension not present in OpenAI's schema. Segments do not include OpenAI's token-level fields (`tokens`, `avg_logprob`, etc.). `language` is the request's `language` value, or `"unknown"` because VibeVoice-ASR does not report the detected language.

**Error Responses:**

| Status | Type | Code | Description |
|--------|------|------|-------------|
| 400 | `invalid_request_error` | `missing_file` | Missing `file` |
| 400 | `invalid_request_error` | `unsupported_file_type` | Unsupported audio extension |
| 400 | `invalid_request_error` | `missing_model` | Missing `model` |
| 400 | `invalid_request_error` | `unsupported_format` | Unsupported `response_format` |
| 400 | `invalid_request_error` | `invalid_temperature` | `temperature` not a number in 0-1 |
| 401 | `authentication_error` | `invalid_api_key` | Invalid API key |
| 503 | `server_error` | - | Task queue is busy |
| 504 | `server_error` | - | Transcription timed out |
| 500 | `server_error` | - | Transcription failed or internal error |

**Examples:**

```bash
curl http://localhost:9527/v1/audio/transcriptions \
  -F file=@meeting.wav \
  -F model=vibevoice-asr \
  -F response_format=verbose_json
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:9527/v1", api_key="unused")
with open("meeting.wav", "rb") as f:
    result = client.audio.transcriptions.create(model="whisper-1", file=f)
print(result.text)
```

### 3. Realtime Transcription (WebSocket)

**WebSocket** `ws://localhost:9527/v1/realtime`

Live transcription of a continuous 24 kHz audio stream using the OpenAI Realtime transcription protocol. The Live Transcription pages (`/live-transcribe`, `/live-transcription`) use this endpoint. The server re-transcribes a rolling window of recent audio with VibeVoice-ASR, commits segments once they stop changing, and streams the results back.

**Connection:**

| Item | Value |
|------|-------|
| Subprotocol | `realtime` (the server echoes it; browsers fail the handshake otherwise) |
| Authentication | `Authorization: Bearer <key>`, or for browsers the extra subprotocol `openai-insecure-api-key.<key>`. Only checked when `OPENAI_COMPAT_API_KEY` is set |
| `project_id` (query, VibeVoice extension) | Save the session into this project's transcription history instead of the standalone one |
| `offloading` (query, VibeVoice extension) | Layer offloading preset: `balanced`, `aggressive` or `extreme` |

The session holds the GPU task slot from the first `session.update` / `input_audio_buffer.append` until it ends, so no other generation, training or transcription can start meanwhile. While it runs, `GET /api/v1/tasks/current` reports it as a `transcription` task with `source: "live"`.

**Client events:**

| Event | Description |
|-------|-------------|
| `session.update` (GA) | `session.type` must be `"transcription"`. `session.audio.input.format` must be `{"type": "audio/pcm", "rate": 24000}` if given. `session.audio.input.transcription.model` selects the precision (`vibevoice-asr` → bf16, `vibevoice-asr-fp8` → float8_e4m3fn, `whisper-1` → bf16, unknown → bf16); `prompt` is used as `context_info`. Replies `session.updated` |
| `transcription_session.update` (beta) | Same, with `input_audio_format: "pcm16"` and `input_audio_transcription: {model, prompt}`. Replies `transcription_session.updated` |
| `input_audio_buffer.append` | `audio`: base64 PCM16, 24 kHz, mono, little-endian |
| `input_audio_buffer.commit` | Transcribe and commit everything received so far |
| `input_audio_buffer.clear` | Drop uncommitted audio. Replies `input_audio_buffer.cleared` |
| `vibevoice.session.finish` (extension) | Stop accepting audio, transcribe the rest, save the record and close the socket |

`turn_detection` is not supported and is reported as `null`; segmentation is done by the rolling window instead.

**Server events:**

| Event | Description |
|-------|-------------|
| `session.created` | Sent on connect (GA shape) |
| `input_audio_buffer.committed` | A new item starts: `item_id`, `previous_item_id` |
| `conversation.item.input_audio_transcription.delta` | Append-only text of the item: the prefix that stayed identical across passes |
| `conversation.item.input_audio_transcription.completed` | Final `transcript` of the item, plus an extension field `segment` (`start_time`, `end_time` in seconds from session start, `speaker_id`, `text`) |
| `vibevoice.transcription.hypothesis` (extension) | Current uncommitted draft: `item_id`, `text`, `segments`, `committed_until` (seconds). Replaced by the next hypothesis; may still change |
| `vibevoice.session.status` (extension) | `status`: `loading_model`, `listening`, `stopping`, `finalizing`, `completed`, `failed`; with `request_id` of the saved transcription, and `reason` (`client_request`, `idle_timeout`, `max_duration`) or `error` |
| `error` | `error.type`, `error.code`, `error.message` |

One item corresponds to one committed ASR segment. A segment is committed when it is unchanged in two consecutive passes, after `STREAMING_ASR_SILENCE_COMMIT_SECONDS` of trailing silence, on `input_audio_buffer.commit`, when the window reaches `STREAMING_ASR_MAX_WINDOW_SECONDS`, or when the session finishes.

**Example session:**
```
→ {"type": "session.update", "session": {"type": "transcription", "audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}, "transcription": {"model": "vibevoice-asr", "prompt": "VibeVoice"}}}}}
← {"type": "session.updated", ...}
← {"type": "vibevoice.session.status", "status": "loading_model", "request_id": "5f0c..."}
← {"type": "vibevoice.session.status", "status": "listening", "request_id": "5f0c..."}
→ {"type": "input_audio_buffer.append", "audio": "<base64 pcm16>"}   (repeated)
← {"type": "vibevoice.transcription.hypothesis", "item_id": null, "text": "Hello every", "segments": [...], "committed_until": 0.0}
← {"type": "input_audio_buffer.committed", "item_id": "item_000001", "previous_item_id": null}
← {"type": "conversation.item.input_audio_transcription.delta", "item_id": "item_000001", "content_index": 0, "delta": "Hello everyone."}
← {"type": "conversation.item.input_audio_transcription.completed", "item_id": "item_000001", "content_index": 0, "transcript": "Hello everyone.", "segment": {"start_time": 0.0, "end_time": 1.8, "speaker_id": 0, "text": "Hello everyone."}}
→ {"type": "vibevoice.session.finish"}
← {"type": "vibevoice.session.status", "status": "stopping", "request_id": "5f0c...", "reason": "client_request"}
← {"type": "vibevoice.session.status", "status": "finalizing", "request_id": "5f0c..."}
← {"type": "vibevoice.session.status", "status": "completed", "request_id": "5f0c..."}
(server closes the socket)
```

**Saving:** every session that received audio is stored as a transcription with `source: "live"` (standalone history, or the project's when `project_id` is given). This also happens when the client disconnects without `vibevoice.session.finish`, after `STREAMING_ASR_IDLE_TIMEOUT_SECONDS` without client messages, or when the session reaches `STREAMING_ASR_MAX_SESSION_SECONDS`.

**Errors and close codes:**

| `error.code` | Close code | When |
|--------------|------------|------|
| `invalid_api_key` | 1008 | API key required and missing or wrong |
| `project_not_found` | 1008 | Unknown `project_id` |
| `invalid_offloading` | 1008 | Unknown `offloading` preset |
| `server_busy` | 1013 at connect, 1000 if the slot was taken before the session started | Another GPU task is running |
| `unsupported_audio_format`, `unsupported_session_type`, `missing_audio`, `invalid_audio`, `invalid_json`, `invalid_message`, `unknown_event` | - | Rejected client event; the session continues |
| `server_error` type | 1000 | Transcription failed; followed by `vibevoice.session.status` `failed` |

**Configuration (environment variables):**

| Variable | Default | Description |
|----------|---------|-------------|
| `STREAMING_ASR_MAX_SESSION_SECONDS` | 1800 | Audio length after which the session finishes (`reason: max_duration`) |
| `STREAMING_ASR_IDLE_TIMEOUT_SECONDS` | 60 | Finish when no client message arrives for this long |
| `STREAMING_ASR_MAX_WINDOW_SECONDS` | 30 | Longest window re-transcribed per pass |
| `STREAMING_ASR_SILENCE_COMMIT_SECONDS` | 1.0 | Trailing silence that commits the window |
| `STREAMING_ASR_SILENCE_RMS` | 0.008 | RMS level below which audio counts as silence |

```python
import asyncio, base64, json, websockets

async def main(pcm16_24k: bytes):
    async with websockets.connect("ws://localhost:9527/v1/realtime", subprotocols=["realtime"]) as ws:
        await ws.send(json.dumps({"type": "session.update", "session": {"type": "transcription"}}))
        for i in range(0, len(pcm16_24k), 4800):
            await ws.send(json.dumps({"type": "input_audio_buffer.append",
                                      "audio": base64.b64encode(pcm16_24k[i:i + 4800]).decode()}))
        await ws.send(json.dumps({"type": "vibevoice.session.finish"}))
        async for message in ws:
            event = json.loads(message)
            if event["type"] == "conversation.item.input_audio_transcription.completed":
                print(event["transcript"])
```

---

### 4. List Models

**GET** `/v1/models`

List available models in OpenAI-compatible format.

**Response (200 OK):**
```json
{
  "object": "list",
  "data": [
    {"id": "tts-1", "object": "model", "created": 0, "owned_by": "vibevoice"},
    {"id": "tts-1-hd", "object": "model", "created": 0, "owned_by": "vibevoice"},
    {"id": "vibevoice-7b", "object": "model", "created": 0, "owned_by": "vibevoice"},
    {"id": "vibevoice-7b-hd", "object": "model", "created": 0, "owned_by": "vibevoice"},
    {"id": "vibevoice-asr", "object": "model", "created": 0, "owned_by": "vibevoice"},
    {"id": "vibevoice-asr-fp8", "object": "model", "created": 0, "owned_by": "vibevoice"},
    {"id": "whisper-1", "object": "model", "created": 0, "owned_by": "vibevoice"}
  ]
}
```

---

# Dataset Management API

## Overview

The Dataset Management API provides endpoints for managing datasets and dataset items. Datasets contain collections of text-audio pairs used for training. All endpoints are project-scoped.

## Endpoints

### 1. List Datasets

**GET** `/api/v1/projects/{project_id}/datasets`

Get all datasets for a project.

**Response (200 OK):**
```json
{
  "datasets": [
    {
      "id": "dataset-id",
      "name": "My Dataset",
      "item_count": 150,
      "created_at": "2025-12-02T10:00:00Z"
    }
  ]
}
```

### 2. Create Dataset

**POST** `/api/v1/projects/{project_id}/datasets`

Create a new dataset.

**Request Body:**
```json
{
  "name": "My New Dataset"
}
```

**Response (201 Created):**
```json
{
  "id": "dataset-id",
  "name": "My New Dataset",
  "item_count": 0,
  "created_at": "2025-12-02T10:00:00Z"
}
```

### 3. Update Dataset

**PUT** `/api/v1/projects/{project_id}/datasets/{dataset_id}`

Update dataset name.

**Request Body:**
```json
{
  "name": "Updated Dataset Name"
}
```

**Response (200 OK):**
```json
{
  "id": "dataset-id",
  "name": "Updated Dataset Name",
  "item_count": 150,
  "created_at": "2025-12-02T10:00:00Z"
}
```

### 4. Delete Dataset

**DELETE** `/api/v1/projects/{project_id}/datasets/{dataset_id}`

Delete a dataset and all its items.

**Response (200 OK):**
```json
{
  "message": "Dataset deleted successfully"
}
```

### 5. Export Dataset

**GET** `/api/v1/projects/{project_id}/datasets/{dataset_id}/export`

Export dataset as ZIP file.

**Response (200 OK):**
- Content-Type: `application/zip`
- Content-Disposition: `attachment; filename="{dataset-name}.zip"`
- Binary ZIP data

**ZIP Structure:**
```
datasets.jsonl              # JSONL file with all items
audio/                      # Audio files directory
  ├── {uuid}.wav
  └── ...
voice_prompts/              # Voice prompt files directory
  ├── {uuid}.wav
  └── ...
```

### 6. Import Dataset

**POST** `/api/v1/projects/{project_id}/datasets/import`

Import dataset from ZIP file.

**Form Data:**
- `file` (required) - ZIP file with dataset structure
- `dataset_id` (required) - Target dataset ID to import into

**Response (200 OK):**
```json
{
  "message": "Dataset imported successfully",
  "item_count": 150
}
```

**Status Codes:**
- `200 OK` - Import successful
- `400 Bad Request` - Invalid ZIP structure or missing dataset
- `404 Not Found` - Project or dataset not found
- `500 Internal Server Error` - Server error

---

## Dataset Item Management

Dataset items contain text-audio pairs with voice prompts for training.

### Base URL

```
http://localhost:9527/api/v1/projects/{project_id}/datasets/{dataset_id}/items
```

## Data Models

### DatasetItem

```json
{
  "text": "string",
  "audio": "filename.wav",
  "voice_prompts": ["prompt1.wav", "prompt2.wav"]
}
```

### Storage Structure

```
workspace/{project_id}/datasets/{dataset_id}/
├── datasets.jsonl              # JSONL file with one item per line
├── audio/                      # Audio files referenced in items
│   └── {uuid}.wav
└── voice_prompts/              # Voice prompt files
    └── {uuid}.wav
```

## API Endpoints

See detailed endpoint documentation below starting with "### 1. List Dataset Items".

---

# Training Management API

## Overview

The Training Management API provides endpoints for managing LoRA fine-tuning training jobs. All endpoints are project-scoped and work with **TrainingState** objects directly. Training and inference share a global task queue (only one runs at a time).

## Base URL

```
http://localhost:9527/api/v1/projects/{project_id}/training
```

## Data Models

### TrainingState

The complete training state object that includes both job metadata and live metrics:

```json
{
  // Job Metadata
  "task_id": "abc123",
  "job_name": "My LoRA Training",
  "project_id": "my-project",
  "config": { /* TrainConfig object */ },
  "created_at": "2025-12-02T10:30:00Z",

  // Progress
  "current_step": 150,
  "estimated_total_steps": 1000,
  "current_epoch": 5,
  "total_epochs": 10,

  // Training Parameters
  "learning_rate": 0.0001,
  "batch_size": 1,
  "accumlate_grad_steps": 16,

  // Loss Metrics
  "current_loss": 0.245,
  "current_diffusion_loss": 0.21,
  "current_ce_loss": 0.035,
  "average_epoch_loss": 0.28,
  "average_epoch_diffusion_loss": 0.24,
  "average_epoch_ce_loss": 0.04,

  // Timing
  "start_time": "2025-12-02T10:35:00Z",
  "current_timestamp": "2025-12-02T10:45:00Z",
  "estimated_total_elpase": 3600.0,
  "latest_epoch_elapsed": 120.5,
  "latest_step_elapsed": 1.2,
  "average_step_time": 1.15,
  "steps_per_second": 0.87,

  // Status
  "status": "Training",  // Prepare, Training, Completed, Failed

  // TensorBoard
  "tensorboard_logdir": "./tensorboard_logs/my_lora_abc123_20251202",

  // Output Files
  "lora_files": ["checkpoint_epoch_5.pt", "checkpoint_epoch_10.pt"],
  "final_lora_file": "final_model.pt"
}
```

### TrainConfig

Training configuration object (matches `vibevoice/training/trainer.py`):

```json
{
  "lora_name": "vibevoice_lora",
  "epochs": 10,
  "batch_size": 1,
  "learning_rate": 0.0001,
  "dataset_path": "workspace/project-id/datasets/dataset-id/datasets.jsonl",
  "output_dir": "./lora_output",
  "multiplier": 1.0,
  "lora_dim": 4,
  "lora_alpha": null,
  "lora_dropout": null,
  "model_path": null,
  "number_of_layers": 0,
  "dtype": "bfloat16",
  "model_config_path": null,
  "optimizer_type": "AdamW8bit",
  "optimizer_args": null,
  "seeds": 42,
  "dataset_repeats": 1,
  "speech_compress_ratio": 3200,
  "semantic_dim": 128,
  "diffusion_loss_weight": 10.4,
  "ce_loss_weight": 0.004,
  "device": "cuda",
  "gradient_accumulation_steps": 16,
  "dataload_workers": 2,
  "save_model_per_num_epoch": 10
}
```

## API Endpoints

### 1. Create Training Job

**POST** `/api/v1/projects/{project_id}/training`

Create and start a new training job. Returns 409 if task manager is busy or job name is duplicate.

**IMPORTANT**: The backend automatically calculates the `output_dir` field based on the job name:
- Path format: `{workspace}/{project}/training/lora_output/{job_name}`
- Job names must be unique within a project
- Any `output_dir` value in the request will be overwritten

**Request Body:**
```json
{
  "job_name": "My LoRA Training",
  "config": {
    "lora_name": "my_lora",
    "epochs": 10,
    "batch_size": 1,
    "learning_rate": 0.0001,
    "dataset_path": "workspace/project-id/datasets/dataset-id/datasets.jsonl",
    "output_dir": "ignored-will-be-overwritten"
  }
}
```

**Response (201 Created):**
```json
{
  "message": "Training started successfully",
  "task_id": "abc123def456",
  "state": { /* TrainingState object */ }
}
```

**Response (409 Conflict - Duplicate Job Name):**
```json
{
  "error": "Conflict",
  "message": "Job name already exists. Please choose a unique name for your training job"
}
```

**Status Codes:**
- `201 Created` - Training job created and started
- `400 Bad Request` - Invalid configuration or missing fields
- `404 Not Found` - Project not found
- `409 Conflict` - Task manager is busy OR job name already exists
- `500 Internal Server Error` - Server error

**Example:**
```bash
curl -X POST "http://localhost:9527/api/v1/projects/my-project/training" \
  -H "Content-Type: application/json" \
  -d '{
    "job_name": "LoRA Fine-tuning v1",
    "config": {
      "lora_name": "my_lora",
      "epochs": 10,
      "batch_size": 1,
      "learning_rate": 0.0001,
      "dataset_path": "workspace/my-project/datasets/dataset-abc/datasets.jsonl"
    }
  }'
```

---

### 2. List Training Jobs

**GET** `/api/v1/projects/{project_id}/training`

Get all training jobs for a project, sorted by created_at (newest first). **This endpoint automatically detects and marks orphaned jobs as Failed.**

**Orphaned Job Detection:**
- If a job has status `Training` but there's no active training task in the task manager
- This happens when the server restarts or crashes during training
- The job status is automatically changed to `Failed` with error message: "Training interrupted (server restart or crash)"
- The metadata is updated immediately and persisted

**Response (200 OK):**
```json
{
  "states": [
    { /* TrainingState object 1 */ },
    { /* TrainingState object 2 */ }
  ],
  "count": 2
}
```

**Status Codes:**
- `200 OK` - Success
- `404 Not Found` - Project not found
- `500 Internal Server Error` - Server error

**Example:**
```bash
curl "http://localhost:9527/api/v1/projects/my-project/training"
```

---

### 3. Get Current Training Job (Project-Specific)

**GET** `/api/v1/projects/{project_id}/training/current`

Get the currently running training job with live metrics from task manager. **Only returns the training job if it belongs to the specified project.**

**Response (200 OK, training in progress for this project):**
```json
{
  "message": "Current training job retrieved successfully",
  "state": { /* Live TrainingState object */ }
}
```

**Response (200 OK, no active training for this project):**
```json
{
  "message": "No active training job at the moment",
  "state": null
}
```

**Status Codes:**
- `200 OK` - Success (state may be null if no active training for this project)
- `404 Not Found` - Project not found
- `500 Internal Server Error` - Server error

**Example:**
```bash
curl "http://localhost:9527/api/v1/projects/my-project/training/current"
```

**Notes:**
- This endpoint reads live state from the training engine in memory, providing real-time metrics updates.
- **Project Filtering**: If a training job is running for a different project, this endpoint returns `null`. Use the unified [Tasks API](#tasks-api-unified) to check if any task is running globally.

---

### 4. Get Specific Training Job

**GET** `/api/v1/projects/{project_id}/training/{job_id}`

Get details of a specific training job by task_id.

**Response (200 OK):**
```json
{
  "state": { /* TrainingState object */ }
}
```

**Status Codes:**
- `200 OK` - Success
- `404 Not Found` - Job or project not found
- `500 Internal Server Error` - Server error

**Example:**
```bash
curl "http://localhost:9527/api/v1/projects/my-project/training/abc123def456"
```

---

### 5. Delete Training Job

**DELETE** `/api/v1/projects/{project_id}/training/{job_id}`

Delete a training job. **Only completed training jobs can be deleted.** This also deletes all associated LoRA files from disk.

**Important Changes:**
- Only jobs with status `Completed` can be deleted
- LoRA files in `output_dir` are automatically deleted using `shutil.rmtree()`
- Failed, Prepare, or Training status jobs cannot be deleted

**Response (200 OK):**
```json
{
  "message": "Training job deleted successfully",
  "job_id": "abc123def456"
}
```

**Response (400 Bad Request - Not Completed):**
```json
{
  "error": "Bad Request",
  "message": "Only completed training jobs can be deleted"
}
```

**Status Codes:**
- `200 OK` - Job and LoRA files deleted successfully
- `400 Bad Request` - Job is not completed (status != Completed)
- `404 Not Found` - Job or project not found
- `500 Internal Server Error` - Server error

**Example:**
```bash
curl -X DELETE "http://localhost:9527/api/v1/projects/my-project/training/abc123def456"
```

**Behavior:**
- Deletes job from `training_history.json`
- Deletes LoRA directory at `{workspace}/{project}/training/lora_output/{job_name}/`
- Only completed jobs can be deleted (prevents accidental deletion of in-progress work)

---

### 6. Download LoRA File

**GET** `/api/v1/projects/{project_id}/training/{job_id}/lora/{filename}`

Download a LoRA checkpoint file from a completed training job.

**Path Parameters:**
- `filename`: The LoRA filename from `TrainingState.lora_files` array (e.g., `checkpoint_epoch_10.pt`)

**Response (200 OK):**
- File download with `Content-Disposition: attachment`
- Content-Type: `application/octet-stream`

**Status Codes:**
- `200 OK` - File download started
- `404 Not Found` - Job not found, file not found, or job not completed
- `500 Internal Server Error` - Server error

**Example:**
```bash
curl -O "http://localhost:9527/api/v1/projects/my-project/training/abc123/lora/checkpoint_epoch_10.pt"
```

**Behavior:**
- Only works for completed training jobs (status == Completed)
- Returns 404 if job is not completed or file doesn't exist
- Downloads from `{workspace}/{project}/training/lora_output/{job_name}/{filename}`

---

### 7. Batch Delete Training Jobs

**POST** `/api/v1/projects/{project_id}/training/batch-delete`

Delete multiple training jobs in one request.

**Request Body:**
```json
{
  "job_ids": ["job1", "job2", "job3"]
}
```

**Response (200 OK):**
```json
{
  "message": "Training jobs deleted successfully",
  "deleted_count": 2,
  "failed_count": 1,
  "deleted_ids": ["job1", "job2"],
  "failed_ids": ["job3"]
}
```

**Status Codes:**
- `200 OK` - Batch deletion completed (check counts for details)
- `400 Bad Request` - Invalid request body
- `404 Not Found` - Project not found
- `500 Internal Server Error` - Server error

**Example:**
```bash
curl -X POST "http://localhost:9527/api/v1/projects/my-project/training/batch-delete" \
  -H "Content-Type: application/json" \
  -d '{"job_ids": ["abc123", "def456", "ghi789"]}'
```

---

## Training Status Values

| Backend Status | Frontend Mapping | Description |
|---------------|-----------------|-------------|
| `Prepare` | `pending` | Job created, waiting to start |
| `Training` | `training` | Training in progress |
| `Completed` | `completed` | Training finished successfully |
| `Failed` | `failed` | Training failed with error |

---

## Architecture Notes

### Shared Task Manager

- Inference and training share a **global task queue** (`task_manager.task.gm`)
- Only **one task** (inference OR training) can run at a time
- Starting a training job when queue is busy returns **409 Conflict**

### Storage Model

- **Single file**: `workspace/{project-id}/training/training_history.json`
- Structure: `{ "task_id": TrainingState, ... }`
- No separate job metadata file - all data in TrainingState

### Live Metrics

**Get current training:**
```python
task = gm.get_current_task()
engine = task.unwrap()
if isinstance(engine, BaseTrainingEngine):
    live_state = engine.get_state()  # Returns live TrainingState
```

**Get historical training:**
```python
# Read from training_history.json
states_meta = json.load(open("training_history.json"))
state = TrainingState.from_dict(states_meta[task_id])
```

### Engine Types

- **TrainingEngine**: Production with GPU, integrates with VibeVoiceTrainer
- **FakeTrainingEngine**: Development/testing, simulates realistic metrics
- Both engines update TrainingState via `TrainingStateWriter`

---

## Translation Support

All messages support bilingual translation (English/Chinese):

**Language Detection:**
- Via `X-Language` header (`en` or `zh`)
- Via `Accept-Language` header
- Defaults to English

**Example:**
```bash
curl -H "X-Language: zh" \
  "http://localhost:9527/api/v1/projects/my-project/training"
```

---

## Error Handling

### Common Error Responses

**Task Manager Busy (409):**
```json
{
  "error": "Conflict",
  "message": "Task manager is busy. Please wait for the current task to complete"
}
```

**Training Job Not Found (404):**
```json
{
  "error": "Not Found",
  "message": "Training job not found"
}
```

**Cannot Delete Running Job (400):**
```json
{
  "error": "Bad Request",
  "message": "Cannot delete a currently running training job"
}
```

---

## Workspace Structure

```
workspace/{project-id}/training/
├── training_history.json  # { "task_id": TrainingState, ... }
└── lora_output/           # LoRA checkpoints (created by trainer)
    └── {lora-name}/
        ├── checkpoint_epoch_5.pt
        ├── checkpoint_epoch_10.pt
        └── final_model.pt
```

---

## Implementation Notes

### Service Layer

Backend follows service layer architecture:

- **API Layer** (`backend/api/training.py`) - HTTP endpoint handlers
- **Service Layer** (`backend/services/training_service.py`) - Business logic with TrainingState
- **Training Layer** (`backend/training/state.py`, `backend/training/engine.py`) - Core training logic

### Key Features

1. **TrainingState-based** - Single dataclass for all job data
2. **Live metrics** - Real-time state from running engine
3. **Atomic operations** - Ensures data consistency
4. **409 on conflict** - Clear feedback when queue is busy
5. **i18n support** - Full bilingual support
6. **FakeTrainingEngine** - Development without GPU

---


