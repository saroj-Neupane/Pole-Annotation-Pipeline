def test_public_pages_render(client):
    for path in ["/", "/about", "/demo", "/health"]:
        response = client.get(path)
        assert response.status_code == 200, path
        assert "text/html" in response.headers["content-type"]


def test_internal_ui_endpoints(client):
    response = client.get("/api/images/list")
    assert response.status_code == 200
    assert sorted(response.json().keys()) == ["midspan", "pole"]


def test_removed_auth_routes_are_gone(client):
    for path in ["/login", "/account", "/integration", "/api/me"]:
        response = client.get(path, follow_redirects=False)
        assert response.status_code == 404, path


def test_public_api_docs_are_disabled(client):
    for path in ["/docs", "/redoc", "/openapi.json"]:
        response = client.get(path, follow_redirects=False)
        assert response.status_code == 404, path


def test_trace_endpoint_validates_input(client):
    response = client.post("/api/trace")
    assert response.status_code == 422

    response = client.post(
        "/demo/predict?pipeline=bogus",
        files={"image": ("a.jpg", b"notanimage", "image/jpeg")},
    )
    assert response.status_code == 400
