// Tenure API: scores one customer's churn risk with the exported logistic model.
//
//   dotnet run --project src/Tenure.Api
//   POST http://localhost:5102/predict   (see requests.http)

using Tenure.Api;
using Tenure.Core;

var builder = WebApplication.CreateBuilder(args);
string modelPath = builder.Configuration["ModelPath"] ?? Path.Combine(AppContext.BaseDirectory, "model.json");
builder.Services.AddSingleton(Predictor.Load(modelPath));
builder.Services.ConfigureHttpJsonOptions(o => o.SerializerOptions.PropertyNamingPolicy = System.Text.Json.JsonNamingPolicy.CamelCase);
builder.Services.AddCors(o => o.AddDefaultPolicy(p => p.AllowAnyOrigin().AllowAnyHeader().WithMethods("GET", "POST")));

var app = builder.Build();
app.UseCors();

app.MapGet("/", () => Results.Ok(new
{
    name = "Tenure churn predictor",
    endpoints = new[] { "GET /health", "POST /predict" },
    docs = "https://github.com/evannwilsonn/tenure",
}));

app.MapGet("/health", (Predictor p) => Results.Ok(new { status = "ok", threshold = Math.Round(p.Threshold, 4) }));

app.MapPost("/predict", (CustomerRequest? body, Predictor predictor) =>
{
    if (body is null) return Results.BadRequest(new { errors = new[] { "Send a JSON body describing one customer." } });
    var (customer, errors) = body.ToCustomer();
    if (customer is not null) errors.AddRange(Predictor.Validate(customer));
    if (errors.Count > 0) return Results.BadRequest(new { errors });
    return Results.Ok(predictor.Predict(customer!));
});

app.Run();

public partial class Program; // lets the tests start the app in memory
