using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using Microsoft.AspNetCore.Mvc.Testing;

namespace Tenure.Tests;

/// <summary>Starts the real API in memory with the trained model.json.</summary>
public class ApiTests : IClassFixture<WebApplicationFactory<Program>>
{
    private readonly HttpClient _client;

    public ApiTests(WebApplicationFactory<Program> factory) => _client = factory.CreateClient();

    private static Dictionary<string, object> Customer(string contract = "Month-to-month", double tenure = 2) => new()
    {
        ["gender"] = "Female", ["seniorCitizen"] = false, ["partner"] = "No", ["dependents"] = "No",
        ["tenure"] = tenure, ["phoneService"] = "Yes", ["multipleLines"] = "No", ["internetService"] = "Fiber optic",
        ["onlineSecurity"] = "No", ["onlineBackup"] = "No", ["deviceProtection"] = "No", ["techSupport"] = "No",
        ["streamingTV"] = "Yes", ["streamingMovies"] = "No", ["contract"] = contract,
        ["paperlessBilling"] = "Yes", ["paymentMethod"] = "Electronic check", ["monthlyCharges"] = 85.5,
    };

    [Fact]
    public async Task Health_reports_the_threshold()
    {
        var body = await _client.GetFromJsonAsync<JsonElement>("/health");
        Assert.Equal("ok", body.GetProperty("status").GetString());
        Assert.InRange(body.GetProperty("threshold").GetDouble(), 0.05, 0.95);
    }

    [Fact]
    public async Task Predict_scores_a_customer_and_explains_it()
    {
        var res = await _client.PostAsJsonAsync("/predict", Customer());
        Assert.Equal(HttpStatusCode.OK, res.StatusCode);
        var body = await res.Content.ReadFromJsonAsync<JsonElement>();
        double p = body.GetProperty("probability").GetDouble();
        Assert.InRange(p, 0, 1);
        Assert.Equal(p >= body.GetProperty("threshold").GetDouble(), body.GetProperty("churn").GetBoolean());
        Assert.True(body.GetProperty("raisesRisk").GetArrayLength() > 0);
    }

    [Fact]
    public async Task A_long_two_year_contract_scores_lower_than_a_new_monthly_one()
    {
        async Task<double> Score(string contract, double tenure)
        {
            var res = await _client.PostAsJsonAsync("/predict", Customer(contract, tenure));
            return (await res.Content.ReadFromJsonAsync<JsonElement>()).GetProperty("probability").GetDouble();
        }
        Assert.True(await Score("Two year", 60) < await Score("Month-to-month", 2));
    }

    [Fact]
    public async Task Missing_and_invalid_fields_are_a_400_with_reasons()
    {
        var missing = await _client.PostAsJsonAsync("/predict", new { gender = "Male" });
        Assert.Equal(HttpStatusCode.BadRequest, missing.StatusCode);
        Assert.Contains("Missing field: tenure", await missing.Content.ReadAsStringAsync());

        var invalid = await _client.PostAsJsonAsync("/predict", Customer(contract: "Weekly"));
        Assert.Equal(HttpStatusCode.BadRequest, invalid.StatusCode);
        Assert.Contains("Contract must be one of", await invalid.Content.ReadAsStringAsync());
    }
}
