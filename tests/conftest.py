import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock

from app.main import app
from app.dependencies import get_supabase_service

@pytest.fixture
def mock_supabase():
    return MagicMock()

@pytest.fixture
def client(mock_supabase):
    """
    Client test API to test supabase.
    """
    app.dependency_overrides[get_supabase_service] = lambda: mock_supabase
    
    with TestClient(app) as c:
        yield c
        
    # Clearing
    app.dependency_overrides.clear()