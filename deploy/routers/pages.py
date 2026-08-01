"""Pages router: /, /about, /demo."""
from fastapi import APIRouter
from fastapi.responses import HTMLResponse

from deploy.utils import load_html_template
from deploy.shared import TEMPLATES_DIR

router = APIRouter(tags=["pages"])


@router.get("/", response_class=HTMLResponse)
async def home_page():
    """Public entry: marketing home page."""
    return load_html_template("landing_page.html", TEMPLATES_DIR)


@router.get("/about", response_class=HTMLResponse)
async def about_page():
    """Product overview, accuracy context, and benchmark figures."""
    return load_html_template("about_page.html", TEMPLATES_DIR)


@router.get("/demo", response_class=HTMLResponse)
async def demo_page():
    """Serve interactive demo page."""
    return load_html_template("demo_page.html", TEMPLATES_DIR)

