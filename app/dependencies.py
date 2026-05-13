from fastapi import Request
from src.services.supabase_api import SupabaseAPIService

def get_supabase_service(request: Request) -> SupabaseAPIService:
    return request.app.state.supabase_svc