"""Security middleware and configuration for the product app."""
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from typing import Callable, Dict, Optional
import logging
from collections import defaultdict
from datetime import datetime, timedelta

from deploy.config import get_settings

logger = logging.getLogger(__name__)


# Rate limiting configuration
class RateLimiter:
    """Simple in-memory rate limiter."""
    
    def __init__(self):
        self.requests: Dict[str, list] = defaultdict(list)
        self.limits = {
            # endpoint_pattern: (requests, seconds)
            "/predict": (100, 3600),  # 100 requests per hour
            # the homepage live showcase spends ~6 predict calls per load, so the
            # demo budget must cover browsing sessions, not just single clicks
            "/demo/predict": (300, 3600),  # 300 requests per hour for demo
        }
    
    def is_allowed(self, client_id: str, endpoint: str) -> tuple[bool, Optional[int]]:
        """
        Check if request is allowed under rate limits.
        
        Args:
            client_id: Client identifier (IP or API key)
            endpoint: Request endpoint
            
        Returns:
            Tuple of (is_allowed, retry_after_seconds)
        """
        # Find matching limit
        limit_config = None
        for pattern, config in self.limits.items():
            if pattern in endpoint:
                limit_config = config
                break
        
        if not limit_config:
            return True, None  # No limit for this endpoint
        
        max_requests, window_seconds = limit_config
        now = datetime.now()
        cutoff = now - timedelta(seconds=window_seconds)
        
        # Clean old requests
        key = f"{client_id}:{endpoint}"
        self.requests[key] = [ts for ts in self.requests[key] if ts > cutoff]
        
        # Check limit
        if len(self.requests[key]) >= max_requests:
            # Calculate retry after
            oldest_request = min(self.requests[key])
            retry_after = int((oldest_request + timedelta(seconds=window_seconds) - now).total_seconds())
            return False, max(retry_after, 1)
        
        # Add current request
        self.requests[key].append(now)
        return True, None


# Global rate limiter instance
rate_limiter = RateLimiter()


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Middleware to enforce rate limiting."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        client_id = request.client.host if request.client else "unknown"
        
        # Check rate limit
        is_allowed, retry_after = rate_limiter.is_allowed(client_id, request.url.path)
        
        if not is_allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded. Please try again later.",
                    "retry_after": retry_after
                },
                headers={"Retry-After": str(retry_after)}
            )
        
        response = await call_next(request)
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Middleware to add security headers."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "img-src 'self' data: blob: https:; "
            "font-src 'self' data:; "
            "connect-src 'self';"
        )
        
        return response


def setup_cors(app, allowed_origins: list = None):
    """
    Setup CORS middleware with secure defaults.
    When allow_credentials=True, wildcard "*" is not allowed by browsers; in production
    we require explicit ALLOWED_ORIGINS (no default "*").
    """
    settings = get_settings()

    if allowed_origins is None:
        allowed_origins = list(settings.allowed_origins)

    if not allowed_origins:
        logger.info("No cross-origin frontend configured; CORS middleware not installed.")
        return

    if "*" in allowed_origins:
        raise RuntimeError(
            "ALLOWED_ORIGINS cannot contain '*'. Configure explicit origins for cross-origin access."
        )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
        max_age=3600,
    )


def setup_security_middleware(app):
    """
    Setup all security middleware.
    
    Args:
        app: FastAPI application
    """
    # Determine if we're in production
    settings = get_settings()
    
    # Security headers
    app.add_middleware(SecurityHeadersMiddleware)

    # Rate limiting
    app.add_middleware(RateLimitMiddleware)

    # CORS
    setup_cors(app)
    
    # Trusted hosts (only in production)
    if settings.is_production and settings.allowed_hosts:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=list(settings.allowed_hosts))
