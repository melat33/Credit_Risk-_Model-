#!/usr/bin/env python
"""
Simple API runner for deadline - WORKING VERSION
"""

import uvicorn

print("🚀 Starting Bati Bank Credit Risk API")
print("📡 http://localhost:8001")
print("📚 Docs: http://localhost:8001/docs")
print("⚡ Fast startup - no reload")

uvicorn.run(
    "src.api.main:app",
    host="0.0.0.0",
    port=8001,
    reload=False,
    log_level="info"
)