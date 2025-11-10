# Frontend - Multimodal Document Chat System

A modern, responsive Next.js-based frontend application for the Multimodal Document Chat System. Users can upload PDF documents, view processing status, and engage in intelligent multimodal conversations with AI powered by extracted document content.

## Project Overview

This frontend application provides an intuitive user interface for document management and AI-powered chat. The application is built with Next.js 14, React 18, and TailwindCSS for a modern, performant user experience.

### Application Pages

- **Home (`/`)** - Document library with upload shortcuts
- **Upload (`/upload`)** - Drag-and-drop file upload interface
- **Chat (`/chat`)** - Multimodal conversation interface
- **Document Details (`/documents/[id]`)** - Individual document view with extracted content

## Tech Stack

### Core Framework
- **Next.js** `14.1.0` - React framework with built-in optimization
- **React** `18.x` - UI library with hooks and concurrent features

### Development & Build
- **TypeScript** `5.x` - Static type checking
- **Node.js** `22-alpine` - JavaScript runtime (Docker)

### Build Optimization
- **Yarn** `latest` - Package manager with lock files
- **Next.js Standalone** - Minimal production builds
- **Image Optimization** - Built-in Next.js image handling

## Setup Instructions

### Prerequisites

- Node.js 18.0+ and npm/yarn
- Backend API running on `http://localhost:8000`

### Option 1: Docker Setup (Recommended)

#### 1. Start from Project Root
```bash
cd /path/to/coding-test-4h
```

#### 2. Start All Services
```bash
docker-compose up -d
```

This will automatically build and start:
- Backend API on `http://localhost:8000`
- Frontend on `http://localhost:3000`

#### 3. Verify Frontend
Open your browser to `http://localhost:3000`

### Option 2: Local Development Setup

#### 1. Navigate to Frontend Directory
```bash
cd frontend
```

#### 2. Install Dependencies
```bash
# Using Yarn (recommended)
yarn install

# Or using npm
npm install
```

#### 3. Configure Environment
```bash
# Create .env.local (if needed for custom backend URL)
echo "NEXT_PUBLIC_BACKEND_URL=http://localhost:8000" > .env.local
```

#### 4. Start Development Server
```bash
yarn dev
# or
npm run dev
```

The application will be available at `http://localhost:3000` with hot reload

### Option 3: Production Build

#### 1. Build the Application
```bash
yarn build
# or
npm run build
```

#### 2. Start Production Server
```bash
yarn start
# or
npm start
```

The application will be available at `http://localhost:3000`

## Environment Variables

Create a `.env.local` file in the `frontend/` directory (optional, uses defaults):

```env
# Backend API URL
# Default: http://localhost:8000
# Use this to point to a different backend instance
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

### Environment Variable Guide

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `NEXT_PUBLIC_BACKEND_URL` | String | `http://localhost:8000` | Backend API base URL |

## Deployment

### Docker Deployment
```bash
# Build image
docker build -t multimodal-chat-frontend:1.0 .

# Run container
docker run -p 3000:3000 -e NEXT_PUBLIC_BACKEND_URL=http://backend:8000 multimodal-chat-frontend:1.0
```

### Vercel Deployment (Recommended for Next.js)
1. Push code to GitHub
2. Connect to Vercel
3. Set environment variables
4. Deploy with one click

### Self-hosted Deployment
1. Build: `npm run build`
2. Copy `.next/`, `node_modules/`, `public/` to server
3. Start: `npm start`
4. Use reverse proxy (nginx, Apache)
