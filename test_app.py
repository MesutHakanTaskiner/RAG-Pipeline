#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Basic tests for the main application.
"""
import pytest
from fastapi.testclient import TestClient
from app import create_app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    app = create_app()
    return TestClient(app)


def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_app_creation():
    """Test that the app can be created successfully."""
    app = create_app()
    assert app is not None
    assert app.title == "RAG API (LangChain + Chroma + OpenAI)"
