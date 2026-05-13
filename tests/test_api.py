def test_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "ok"
    assert "timestamp" in data

def test_get_recent_predictions_success(client, mock_supabase):

    mock_supabase.get_latest_predictions.return_value = [
        {
            "predicted_at": "2026-04-13T10:00:00Z",
            "forecast_horizon": "2026-05-13T11:00:00Z",
            "predicted_value": 125.4
        },
        {
            "predicted_at": "2026-04-13T10:00:00Z",
            "forecast_horizon": "2026-05-13T12:00:00Z",
            "predicted_value": 130.2
        }
    ]

    # Call HTTP
    response = client.get("/solar/predictions/latest")
    assert response.status_code == 200
    data = response.json()

    # Parameters
    assert len(data["predictions"]) == 2
    assert data["predicted_at"] == "2026-04-13T10:00:00Z"
    assert data["predictions"][0]["predicted_value"] == 125.4
    
    mock_supabase.get_latest_predictions.assert_called_once_with(
        table_name="predictions"
    )

def test_get_latest_predictions_no_data(client, mock_supabase):
    mock_supabase.get_latest_predictions.return_value = []

    response = client.get("/solar/predictions/latest")

    assert response.status_code == 404
    assert response.json() == {
        "detail": "No data available"
    }