# API Validation — Hardening `/predict`

**Branch:** `feature/sentiment-api`

---

# Part 1 — Concepts

## Why input validation belongs at the API boundary

Input validation should happen before the request reaches the model. The API checks whether the input is valid, such as making sure it is a non-empty string and not too long. If invalid data is rejected early, the model only receives valid input, making the service more reliable and preventing unexpected errors.

---

## Pydantic models for request validation

FastAPI uses Pydantic models to define the expected request format.

```python
class PredictRequest(BaseModel):
    text: str
```

Pydantic automatically checks that the request contains a valid string. Additional validation is added with a `@field_validator` to reject empty or whitespace-only text and enforce the maximum input length before the request reaches the prediction function.

---

## Client errors (4xx) vs Server errors (5xx)

A **4xx** error means there is a problem with the client's request, such as sending an empty string or the wrong data type. The client can fix the request and try again.

A **5xx** error means something failed on the server, such as a missing model file or an unexpected error during prediction. In this case, the problem is with the service, not the client.

---

## Why clear error messages matter

Returning a clear error message helps the user understand what went wrong instead of returning an incorrect prediction or a confusing error. For example, if the input is empty, the API returns a **422** response with a message explaining the problem rather than trying to make a prediction.

---

# Part 2 — Hardening Implemented

The `/predict` endpoint was improved by adding:

* Validation to reject empty or whitespace-only text.
* A maximum input length of **1000 characters**.
* Automatic rejection of invalid data types using Pydantic.
* A `low_confidence` flag that becomes `true` when the prediction confidence is below **0.55**, while still returning the predicted label.
* A custom validation handler that returns simple and readable error messages.

---

# Test Results

The API was tested by sending HTTP requests to the running FastAPI service. The expected and actual results are shown below.

| Input                             | Test Case       | Expected | Actual                                                                    |
| --------------------------------- | --------------- | -------- | ------------------------------------------------------------------------- |
| `{"text":"The product is great"}` | Valid input     | 200      | ✅ 200 – `{"label":"Positive","confidence":0.6647,"low_confidence":false}` |
| `{"text":""}`                     | Empty string    | 422      | ✅ 422 – `text must not be empty or whitespace-only`                       |
| `{"text":"   "}`                  | Whitespace only | 422      | ✅ 422 – `text must not be empty or whitespace-only`                       |
| `1500-character string`           | Too long        | 422      | ✅ 422 – maximum length exceeded                                           |
| `{"text":12345}`                  | Wrong type      | 422      | ✅ 422 – input should be a valid string                                    |
| `{}`                              | Missing field   | 422      | ✅ 422 – field required                                                    |
| `{"text":"هذا المنتج رائع جدا"}`  | Non-English     | 200      | ✅ 200 – `{"label":"Neutral","confidence":0.616,"low_confidence":false}`   |

All test cases produced the expected status codes.

