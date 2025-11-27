# CV Screener Web UI

Next.js 16 App for the recruiter chat experience, styled with Tailwind 4 and shadcn/ui.

## Run locally

- Install deps with `pnpm install` (Corepack is enabled in the Dockerfile if you prefer `corepack enable`).
- Set `NEXT_PUBLIC_API_URL` for the FastAPI backend (defaults to `http://localhost:8000`).
- Start dev server: `pnpm dev` at http://localhost:3000; lint/build with `pnpm lint` / `pnpm build`, serve production
  with `pnpm start`.

## Docker

- From the repo root run `docker compose up web` to build via `web/Dockerfile` (standalone output on port 3000).

## Notes

- Chat entry point is `app/page.tsx`; it posts to `${NEXT_PUBLIC_API_URL}/query` and renders source-backed answers.
- UI primitives live under `components/ui/`; global styles and Tailwind config are in `styles/` and
  `postcss.config.mjs`.
