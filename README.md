# ApplyStack AI

An LLM-powered, review-first job application operations platform. ApplyStack AI
uses verified profile data to prepare portal applications, surface unanswered
questions, track application runs, and provide cost telemetry while keeping final
submission under human control.

## Architecture

- `frontend/`: React and TypeScript operations console, built into an Nginx image.
- `backend/`: FastAPI application, browser automation services, persistent queue,
  and Podman Compose production stack.

## Run With Podman

Create `backend/.env` from `backend/.env.example`, configure the required
provider values, then run:

```bash
podman-compose -f backend/podman-compose.yml up --build
```

The UI is available at `http://localhost:8080`; the API health endpoint is
`http://localhost:8000/api/v1/health`.

## Safety

The platform prepares applications for human review. It never performs a final
application submission.
