# Docker Rebuild Workflow for Code Updates

## Problem

Since this Dockerfile clones code from GitHub (no local context dependency), Docker cannot detect when you've pushed new code. Running `docker compose build` does nothing because no local files changed.

## Solution: Cache Busting with Build Args

We use a `CACHE_BUST` build argument that changes each build to force Docker to invalidate cache from the git clone step onwards.

### How It Works

1. **Dockerfile** (`Dockerfile:37-43`): Added `CACHE_BUST` arg in Stage 2 (before git clone)
   ```dockerfile
   ARG CACHE_BUST=unknown
   RUN echo "Cache bust: ${CACHE_BUST}"  # Invalidates cache when value changes
   ```

2. **Compose file** (`compose.yml:12,23`): Passes `CACHE_BUST` from environment
   ```yaml
   args:
     CACHE_BUST: ${CACHE_BUST:-unknown}
   ```

3. **Helper script** (`rebuild.sh`): Automates the workflow
   ```bash
   CACHE_BUST=$(date +%Y%m%d-%H%M%S)
   export CACHE_BUST
   docker compose build backend-builder
   docker compose build vibevoice
   ```

## Usage

### Option 1: Use the Helper Script (Recommended)

```bash
./rebuild.sh
```

This automatically:
- Generates a timestamp-based cache bust value
- Rebuilds backend-builder stage (git clone + frontend + python venv)
- Rebuilds final vibevoice image
- **Skips** model download stage (cached, ~3-4GB)

### Option 2: Manual Build with Cache Bust

```bash
# Set cache bust value (any unique string works)
export CACHE_BUST=$(date +%Y%m%d-%H%M%S)

# Or use git commit hash for traceability
export CACHE_BUST=$(git rev-parse --short HEAD)

# Rebuild
docker compose build backend-builder
docker compose build vibevoice
```

### Option 3: Force Full Rebuild (Slow)

If you need to rebuild everything including models:

```bash
docker compose build --no-cache
```

**Warning**: This re-downloads 3-4GB of model files from HuggingFace!

## What Gets Rebuilt?

### With `./rebuild.sh` (Fast - 5-10 minutes)

✅ **Rebuilt:**
- Stage 2: Git clone from GitHub
- Stage 2: Frontend npm install + build
- Stage 3: Python venv + pip install
- Stage 4: Final assembly

💾 **Cached:**
- Stage 1: Model download (3-4GB)
- Stage 4: System packages

### With `--no-cache` (Slow - 20-30 minutes)

🔄 **Rebuilt:**
- Everything including model download

## Build Time Comparison

| Method | Time | Use Case |
|--------|------|----------|
| `./rebuild.sh` | 5-10 min | Code updates (normal workflow) |
| `--no-cache` | 20-30 min | First build or model updates |
| No cache bust | 0 sec | Nothing happens (problem!) |

## Advanced: Custom Cache Bust Values

You can use custom values for better traceability:

```bash
# Use git commit hash
export CACHE_BUST=$(git rev-parse --short HEAD)
docker compose build backend-builder vibevoice

# Use version tag
export CACHE_BUST=v1.2.3
docker compose build backend-builder vibevoice

# Use build number (CI/CD)
export CACHE_BUST=build-${BUILD_NUMBER}
docker compose build backend-builder vibevoice
```

## Troubleshooting

### Q: Build says "cached" even with rebuild.sh
**A:** Make sure the script has execute permissions:
```bash
chmod +x rebuild.sh
```

### Q: Can I skip backend-builder and only rebuild vibevoice?
**A:** No, because vibevoice depends on backend-builder. You must rebuild both in order.

### Q: How do I verify the latest code is in the image?
**A:** Check the git commit hash inside the container:
```bash
docker compose run --rm vibevoice cat backend/version.txt
```

### Q: Can I use this for local development instead of GitHub?
**A:** Yes! Modify Stage 2 to use `COPY . /build/vibevoice` instead of `git clone`. Then your local file changes will trigger rebuilds automatically.

## Performance Tips

1. **Use buildkit** for parallel stage builds:
   ```bash
   DOCKER_BUILDKIT=1 ./rebuild.sh
   ```

2. **Build only what you need**:
   ```bash
   # If you only changed backend code (not frontend)
   # Still use rebuild.sh - npm install uses cache if package.json unchanged
   ./rebuild.sh
   ```

3. **Prune old images** to save disk space:
   ```bash
   docker image prune -a
   ```

## See Also

- `Dockerfile` - Multi-stage build definition
- `compose.yml` - Service configuration
- `rebuild.sh` - Automated rebuild script
