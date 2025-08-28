# RAG Pipeline - Testing Guide

## Overview
This document provides comprehensive instructions for running tests and using the RAG (Retrieval-Augmented Generation) Pipeline application.

## Project Structure

```
RAG-Pipeline/
├── app.py                      # Main FastAPI application
├── config/
│   ├── __init__.py
│   └── settings.py            # Configuration management
├── models/
│   ├── __init__.py
│   └── schemas.py             # Pydantic models
├── services/
│   ├── __init__.py
│   ├── vector_store.py        # Vector store operations
│   ├── llm.py                 # Language model operations
│   ├── mmr.py                 # Maximal Marginal Relevance
│   ├── query_decomposer.py    # Query decomposition
│   └── reasoning_agent.py     # Multi-step reasoning
├── utils/
│   ├── __init__.py
│   └── text_processing.py     # Text processing utilities
├── api/
│   ├── __init__.py
│   └── routes.py              # API route handlers
├── tests/                     # Test suite
│   ├── __init__.py
│   ├── test_app.py           # Application tests
│   ├── test_config.py        # Configuration tests
│   ├── test_models.py        # Model validation tests
│   └── test_utils.py         # Utility function tests
├── data/                     # Data directory
│   ├── processed/
│   └── raw/
├── embed/
│   └── embedding.py
├── .env.example              # Environment variables template
├── .gitignore
└── README.md
```

## Prerequisites

### 1. Python Environment
- Python 3.11 or higher
- Virtual environment (recommended)

### 2. Dependencies
Install required packages:
```bash
pip install -U fastapi uvicorn python-dotenv langchain-openai langchain-community chromadb langchain-core pydantic pytest httpx
```

### 3. Environment Setup
Create a `.env` file based on `.env.example`:
```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-your-openai-api-key-here
LLM_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-large
EMBEDDING_DIM=3072

# Vector Store Configuration
CHROMA_PERSIST_DIR=.chroma/ntt_reports_openai_v1
COLLECTION_NAME=ntt_reports_openai_v1

# Retrieval Configuration
RETRIEVAL_K=12
CONTEXT_K=6
CONTEXT_CHAR_LIMIT=1200
```

## Running Tests

### Quick Test Run
Run all tests:
```bash
python -m pytest tests/
```

### Verbose Test Output
Run tests with detailed output:
```bash
python -m pytest tests/ -v
```

### Run Specific Test Files
```bash
# Test application functionality
python -m pytest tests/test_app.py -v

# Test configuration
python -m pytest tests/test_config.py -v

# Test Pydantic models
python -m pytest tests/test_models.py -v

# Test utility functions
python -m pytest tests/test_utils.py -v
```

### Test Coverage
Run tests with coverage report:
```bash
python -m pytest tests/ --cov=. --cov-report=html
```

### Test Categories

#### 1. Application Tests (`test_app.py`)
- **test_health_endpoint**: Tests the `/health` endpoint
- **test_app_creation**: Verifies FastAPI app creation

#### 2. Configuration Tests (`test_config.py`)
- **test_settings_from_env**: Tests environment variable loading
- **test_settings_defaults**: Verifies default configuration values
- **test_get_settings**: Tests settings retrieval function
- **test_settings_validation**: Validates configuration constraints

#### 3. Model Tests (`test_models.py`)
- **test_ask_request_valid**: Tests valid request model creation
- **test_ask_request_validation_errors**: Tests request validation errors
- **test_source_model**: Tests source document model
- **test_source_page_validation**: Tests page range validation
- **test_ask_response_model**: Tests response model creation
- **test_health_response_model**: Tests health response model
- **test_reasoning_step_response**: Tests reasoning step model
- **test_sub_question_response**: Tests sub-question response model
- **test_decomposition_response**: Tests query decomposition model

#### 4. Utility Tests (`test_utils.py`)
- **test_trim_text**: Tests text trimming functionality
- **test_format_context_block**: Tests context formatting
- **test_extract_metadata_safely**: Tests safe metadata extraction
- **test_validate_text_length**: Tests text length validation
- Various edge case tests for utility functions

## Running the Application

### 1. Development Server
Start the FastAPI development server:
```bash
uvicorn app:app --reload --port 8080
```

### 2. Production Server
For production deployment:
```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --workers 4
```

### 3. API Documentation
Once running, access the interactive API documentation:
- Swagger UI: http://localhost:8080/docs
- ReDoc: http://localhost:8080/redoc

## API Endpoints

### Health Check
```bash
GET /health
```
Returns system status and configuration information.

### Ask Question
```bash
POST /ask
```
Submit questions to the RAG system.

Example request:
```json
{
  "question": "What are the main sustainability initiatives?",
  "year_from": 2022,
  "year_to": 2024,
  "top_k": 10,
  "use_mmr": true,
  "mmr_lambda": 0.7,
  "use_reasoning": true,
  "show_reasoning_trace": true
}
```

## Testing Best Practices

### 1. Environment Isolation
Tests use environment variable manipulation to ensure isolation:
```python
# Tests clean up environment variables
original_values = {}
for var in env_vars_to_clear:
    if var in os.environ:
        original_values[var] = os.environ[var]
        del os.environ[var]
```

### 2. Mock External Dependencies
Tests avoid external API calls by mocking services when needed.

### 3. Comprehensive Validation Testing
All Pydantic models are tested for:
- Valid input scenarios
- Invalid input validation
- Edge cases and boundary conditions

### 4. Error Handling
Tests verify proper error handling and HTTP status codes.

## Troubleshooting

### Common Issues

#### 1. Import Errors
If you encounter import errors, ensure you're running tests from the project root:
```bash
cd /path/to/RAG-Pipeline
python -m pytest tests/
```

#### 2. Environment Variables
Ensure your `.env` file is properly configured with valid API keys.

#### 3. Dependencies
If tests fail due to missing dependencies:
```bash
pip install -r requirements.txt  # if available
# or install manually:
pip install fastapi uvicorn pytest httpx pydantic langchain-openai
```

#### 4. Pydantic Warnings
The project uses Pydantic V2. If you see deprecation warnings, ensure you have the latest version:
```bash
pip install -U pydantic
```

## Continuous Integration

### GitHub Actions Example
```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: 3.11
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest
    - name: Run tests
      run: pytest tests/
```

## Performance Testing

### Load Testing with pytest-benchmark
```bash
pip install pytest-benchmark
python -m pytest tests/ --benchmark-only
```

### Memory Usage Testing
```bash
pip install pytest-memray
python -m pytest tests/ --memray
```

## Test Data

### Mock Data
Tests use mock data to avoid dependencies on external services:
- Mock documents with metadata
- Sample configuration values
- Predefined test responses

### Test Fixtures
Common test fixtures are available in `tests/conftest.py` (if created):
```python
@pytest.fixture
def sample_document():
    return Document(
        page_content="Sample content",
        metadata={"doc_id": "test", "year": 2023}
    )
```

## Contributing

### Adding New Tests
1. Create test files in the `tests/` directory
2. Follow naming convention: `test_*.py`
3. Use descriptive test function names: `test_feature_scenario`
4. Include docstrings explaining test purpose
5. Test both success and failure scenarios

### Test Guidelines
- Each test should be independent
- Use appropriate assertions
- Clean up resources after tests
- Mock external dependencies
- Test edge cases and error conditions

## Support

For issues related to testing:
1. Check this documentation
2. Review test output for specific error messages
3. Ensure all dependencies are installed
4. Verify environment configuration
5. Check Python version compatibility

## Summary

The RAG Pipeline now includes a comprehensive test suite with 23 tests covering:
- ✅ Application functionality
- ✅ Configuration management
- ✅ Data model validation
- ✅ Utility functions
- ✅ Error handling
- ✅ Edge cases

All tests pass successfully and the project is ready for development and deployment.
